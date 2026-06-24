"""
Cross-Encoder L1 vs L1Tilde — 不冻结 Encoder（端到端训练）

与 cross_encoder_correct.py 的唯一区别：Encoder 参数 requires_grad=True
整个 GNN+MLP（或 Embedding+MLP）一起训练 50 epochs。

用法:
  python ce_unfrozen.py --data_dir data/OSM_Chengdu --model_class sage \
      --gnn_ckpt <path> --epochs 50 --r 2 --s 62
"""
import numpy as np, torch, torch.nn as nn, time, sys, os, argparse
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from utils.data_utils import load_graph, get_edge_attributes, get_node_attributes
from utils.torch_utils import read_query_file
from models.rgnndist2vec import RGNNdist2vec

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/OSM_Chengdu')
parser.add_argument('--model_class', required=True, choices=['sage','gat','gcn','aneda','rne'])
parser.add_argument('--gnn_ckpt', required=True)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--s', type=int, default=62)
args = parser.parse_args()

DATA_DIR = args.data_dir; CITY = DATA_DIR.split('/')[-1]
HIDDEN = 512; EMB_DIM = 64; EPOCHS = args.epochs; LR = args.lr; R, S = args.r, args.s

# Load data
G = load_graph(dir_name=DATA_DIR, force_shift=0)
edge_attr = get_edge_attributes(G)
node_attr = get_node_attributes(G)
train_q = np.array(read_query_file(f"{DATA_DIR}/random_500k/{CITY}_train.queries"))
test_q  = np.array(read_query_file(f"{DATA_DIR}/random_500k/{CITY}_test.queries"))
max_dist = train_q[:, 2].max()

# ---- GNN-based models (SAGE/GAT/GCN) ----
if args.model_class in ('sage','gat','gcn'):
    # Load Bi-Encoder as starting point (NOT frozen)
    model = RGNNdist2vec(n_input=2, n_hidden_1=HIDDEN, n_hidden_2=EMB_DIM,
                          layer_type=args.model_class, node_attributes=node_attr,
                          edge_attributes=edge_attr, max_distance=max_dist,
                          disable_edge_weight=True)
    ckpt = torch.load(args.gnn_ckpt, map_location='cuda', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    ms = model.state_dict()
    fd = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    model.load_state_dict(fd, strict=False)
    model.cuda()

    # All encoder params trainable
    for p in model.parameters(): p.requires_grad = True

    # Pre-compute features once per epoch in training loop
    coord_mean, coord_std = node_attr.mean(axis=0), node_attr.std(axis=0)
    coords_norm = (node_attr - coord_mean) / coord_std
    base_feat = torch.from_numpy(coords_norm).float().cuda()

    # MLP head
    d_in = (EMB_DIM + 2) * 2  # 132
    mlp = nn.Sequential(
        nn.Linear(d_in, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(HIDDEN, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(HIDDEN, EMB_DIM * 2),
    ).cuda()

    def encode_full_graph():
        # 不冻结: GNN 正常前传，梯度可回传
        return model.encode(model.geometric_data.x.cuda(),
                           model.geometric_data.edge_index.cuda())

    def forward_pair(embeddings, u, v, mode):
        feats = torch.cat([embeddings, base_feat], dim=1)
        fu, fv = feats[u], feats[v]
        out = mlp(torch.cat([fu, fv], dim=1))
        y_o, y_d = out[:, :EMB_DIM], out[:, EMB_DIM:]
        if mode == 'l1':
            return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True)
        else:
            sym = torch.abs(y_d[:, :R] - y_o[:, :R]).sum(dim=1, keepdim=True)
            asym = (y_d[:, R:R+S] - y_o[:, R:R+S]).sum(dim=1, keepdim=True)
            return sym + asym

# ---- Embedding-based models (ANEDA/RNE) ----
elif args.model_class in ('aneda', 'rne'):
    ckpt = torch.load(args.gnn_ckpt, map_location='cuda', weights_only=False)
    sd = ckpt['model_state_dict']
    emb_key = [k for k in sd if 'embedding.weight' in k][0]
    emb_w = sd[emb_key].clone().cuda()
    N = emb_w.shape[0]

    # Unfrozen embedding
    embedding = nn.Embedding(N, EMB_DIM).cuda()
    embedding.weight.data.copy_(emb_w)

    d_in = EMB_DIM * 2
    mlp = nn.Sequential(
        nn.Linear(d_in, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(HIDDEN, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(HIDDEN, EMB_DIM * 2),
    ).cuda()

    # All params trainable
    for p in embedding.parameters(): p.requires_grad = True

    def forward_pair(embeddings, u, v, mode):
        eu, ev = embedding(u), embedding(v)
        out = mlp(torch.cat([eu, ev], dim=1))
        y_o, y_d = out[:, :EMB_DIM], out[:, EMB_DIM:]
        if mode == 'l1':
            return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True)
        else:
            sym = torch.abs(y_d[:, :R] - y_o[:, :R]).sum(dim=1, keepdim=True)
            asym = (y_d[:, R:R+S] - y_o[:, R:R+S]).sum(dim=1, keepdim=True)
            return sym + asym

# ---- Training ----
def train(mode, Xt_u, Xt_v, yt, Xv_u, Xv_v, yv):
    is_gnn = (args.model_class in ('sage','gat','gcn'))
    opt = torch.optim.Adam(
        list(mlp.parameters()) +
        (list(model.parameters()) if is_gnn
         else list(embedding.parameters())),
        lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    Xt_u, Xt_v = Xt_u.cuda(), Xt_v.cuda(); yt = yt.cuda()
    Xv_u, Xv_v = Xv_u.cuda(), Xv_v.cuda(); yv = yv.cuda()
    best = float('inf')

    for ep_i in range(EPOCHS):
        # 每 epoch 编码一次全图（GNN不冻结，梯度可回传）
        if is_gnn:
            emb_all = encode_full_graph()
        else:
            # Embedding-based: always have grads
            emb_all = None
        perm = torch.randperm(len(yt))
        for b in range(0, len(yt), 4096):
            idx = perm[b:b+4096]
            opt.zero_grad()
            pred = forward_pair(emb_all if is_gnn else None,
                               Xt_u[idx], Xt_v[idx], mode)
            loss = nn.SmoothL1Loss()(pred, yt[idx] / max_dist)
            loss.backward(); opt.step()

        # Validation: 用 no_grad（不需要梯度）
        with torch.no_grad():
            if is_gnn:
                model.eval()
                emb_val = encode_full_graph()
            pred = forward_pair(emb_val if is_gnn else None,
                               Xv_u, Xv_v, mode)
            mre = (torch.abs(pred * max_dist - yv) / (yv + 1e-8)).mean().item()
            sch.step(mre)
            if mre < best: best = mre
            if is_gnn: model.train()

    return best * 100

# Prepare data tensors
u_train = torch.from_numpy(train_q[:, 0].astype(int)).long()
v_train = torch.from_numpy(train_q[:, 1].astype(int)).long()
y_train = torch.from_numpy(train_q[:, 2].astype(np.float32)).reshape(-1, 1)
u_test = torch.from_numpy(test_q[:, 0].astype(int)).long()
v_test = torch.from_numpy(test_q[:, 1].astype(int)).long()
y_test = torch.from_numpy(test_q[:, 2].astype(np.float32)).reshape(-1, 1)

print(f"\n{'='*60}")
print(f"Unfrozen CE L1 vs L1Tilde ({CITY}, {args.model_class}, {EPOCHS} epochs)")
print(f"{'='*60}")
for mode in ['l1', 'l1tilde']:
    t0 = time.time()
    mre = train(mode, u_train, v_train, y_train, u_test, v_test, y_test)
    print(f"  {mode:>8s}: Test MRE = {mre:.2f}%  ({time.time()-t0:.0f}s)")
