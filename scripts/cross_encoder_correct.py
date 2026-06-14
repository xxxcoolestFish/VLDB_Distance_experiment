"""
Cross-Encoder L1 vs L1Tilde — 正确方法（两阶段训练）

阶段1: 加载预训练GNN → 编码所有节点 → 冻结特征
阶段2: concat(feat_u, feat_v) → MLP → y_o, y_d → L1/L1Tilde (只训MLP)

对应导师 ablation_study/scripts/cross_encoder_test.py 的方法。
"""
import numpy as np, torch, torch.nn as nn, time, sys, argparse
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, '.')
from utils.data_utils import load_graph, get_edge_attributes, get_node_attributes
from utils.torch_utils import read_query_file
from models.rgnndist2vec import RGNNdist2vec

torch.manual_seed(42)

# ---- Config ----
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/OSM_Chengdu')
parser.add_argument('--gnn_ckpt', required=True, help='Path to GNN checkpoint')
parser.add_argument('--gnn_layer', default='sage', choices=['sage','gat','gcn'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--s', type=int, default=62)
args = parser.parse_args()

DATA_DIR = args.data_dir
CITY = DATA_DIR.split('/')[-1]
GNN_CKPT = args.gnn_ckpt
HIDDEN = 512; EMB_DIM = 64; EPOCHS = args.epochs
LR = args.lr; R, S = args.r, args.s

# ---- Load data ----
G = load_graph(dir_name=DATA_DIR, force_shift=0)
edge_attr = get_edge_attributes(G)
node_attr = get_node_attributes(G)
print(f"Graph: {node_attr.shape[0]} nodes, {edge_attr.shape[0]} edges")

train_q = np.array(read_query_file(f"{DATA_DIR}/random_500k/{CITY}_train.queries"))
test_q  = np.array(read_query_file(f"{DATA_DIR}/random_500k/{CITY}_test.queries"))
max_dist = train_q[:, 2].max()
print(f"Queries: {len(train_q)} train, {len(test_q)} test, max dist={max_dist:.0f}m")

# ---- Extract frozen GNN features ----
model = RGNNdist2vec(n_input=2, n_hidden_1=HIDDEN, n_hidden_2=EMB_DIM,
                      layer_type=args.gnn_layer, node_attributes=node_attr,
                      edge_attributes=edge_attr, max_distance=max_dist,
                      disable_edge_weight=True)
ckpt = torch.load(GNN_CKPT, map_location='cuda', weights_only=False)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
# GAT checkpoints may have edge_weight; strip if model doesn't expect it
model_state = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(filtered_dict, strict=False)
model.cuda(); model.eval()
with torch.no_grad():
    emb = model.encode(model.geometric_data.x.cuda(), model.geometric_data.edge_index.cuda()).cpu().numpy()
print(f"GNN embeddings: {emb.shape}")

# Normalize coordinates + concat with GNN features
coord_mean, coord_std = node_attr.mean(axis=0), node_attr.std(axis=0)
coords_norm = (node_attr - coord_mean) / coord_std
feat = np.hstack([emb, coords_norm]).astype(np.float32)  # (N, 64+2=66)
print(f"Cross-Encoder input dim: {feat.shape[1]}*2 = {feat.shape[1]*2}")

def make_data(q):
    u, v = q[:, 0].astype(int), q[:, 1].astype(int)
    X = np.hstack([feat[u], feat[v]]).astype(np.float32)
    y = (q[:, 2].astype(np.float32) / max_dist).reshape(-1, 1)
    return X, y

X_train, y_train = make_data(train_q)
X_test,  y_test  = make_data(test_q)

# ---- Cross-Encoder Model ----
class CrossEncoder(nn.Module):
    def __init__(self, d_in, mode='l1', r=R, s=S, emb_dim=EMB_DIM):
        super().__init__()
        self.mode, self.r, self.s = mode, r, s
        self.out_dim = emb_dim * 2  # y_o + y_d
        self.net = nn.Sequential(
            nn.Linear(d_in, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN, self.out_dim),
        )

    def forward(self, x):
        out = self.net(x)  # (B, emb_dim*2)
        y_o, y_d = out[:, :self.out_dim//2], out[:, self.out_dim//2:]
        if self.mode == 'l1':
            return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True)
        else:  # l1tilde
            r, s = self.r, self.out_dim//2 - self.r
            sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(dim=1, keepdim=True)
            asym = (y_d[:, r:r+s] - y_o[:, r:r+s]).sum(dim=1, keepdim=True)
            return sym + asym

def train(mode, Xt, yt, Xv, yv):
    m = CrossEncoder(Xt.shape[1], mode=mode).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt)),
                    batch_size=4096, shuffle=True)
    Xvt, yvt = torch.from_numpy(Xv).cuda(), torch.from_numpy(yv).cuda()
    best, best_ep = float('inf'), 0
    for ep_i in range(EPOCHS):
        m.train()
        for bx, by in dl:
            bx, by = bx.cuda(), by.cuda()
            opt.zero_grad(); nn.SmoothL1Loss()(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mre = (torch.abs(m(Xvt) - yvt) / (yvt + 1e-8)).mean().item()
            sch.step(mre)
            if mre < best: best, best_ep = mre, ep_i + 1
    return best * 100, best_ep

# ---- Run ----
print(f"\n{'='*60}")
print(f"Cross-Encoder L1 vs L1Tilde ({CITY}, GNN={args.gnn_layer}, {EPOCHS} epochs)")
print(f"{'='*60}")
for mode in ['l1', 'l1tilde']:
    t0 = time.time()
    mre, bep = train(mode, X_train, y_train, X_test, y_test)
    print(f"  {mode:>8s}: Test MRE = {mre:.2f}%  (best epoch {bep}, {time.time()-t0:.0f}s)")
