"""
Cross-Encoder for Embedding-based baselines (RNE, ANEDA, NDist2Vec, VDist2Vec).
Loads frozen embeddings → concat(feat_u, feat_v) → MLP → L1/L1Tilde.
"""
import numpy as np, torch, torch.nn as nn, time, sys, os, argparse
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from utils.data_utils import load_graph, read_query_file

torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/OSM_Chengdu')
parser.add_argument('--model_class', required=True)
parser.add_argument('--ckpt', required=True)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--s', type=int, default=62)
args = parser.parse_args()

DATA_DIR = args.data_dir; CITY = DATA_DIR.split('/')[-1]
EPOCHS, LR, R, S = args.epochs, args.lr, args.r, args.s
HIDDEN = 512; EMB_DIM = 64

# Load data
G = load_graph(dir_name=DATA_DIR, force_shift=0)
train_q = np.array(read_query_file(f"{DATA_DIR}/random_500k/{CITY}_train.queries"))
test_q  = np.array(read_query_file(f"{DATA_DIR}/random_500k/{CITY}_test.queries"))
max_dist = train_q[:, 2].max()

# Load model and extract embeddings
print(f"Loading {args.model_class} from {args.ckpt}...")
ckpt = torch.load(args.ckpt, map_location='cuda', weights_only=False)
sd = ckpt['model_state_dict']

# Extract embedding weight (handle both direct embedding and GNN-based)
emb_key = None
for k in sd:
    if 'embedding.weight' in k: emb_key = k; break
if emb_key:
    emb = sd[emb_key].cpu().numpy()
    print(f"Embedding: {emb.shape}")
else:
    # Try to use the model to encode
    raise ValueError("No embedding.weight found in checkpoint")

feat = emb.astype(np.float32)
print(f"Cross-Encoder input dim: {feat.shape[1]}*2 = {feat.shape[1]*2}")

def make_data(q):
    u, v = q[:, 0].astype(int), q[:, 1].astype(int)
    X = np.hstack([feat[u], feat[v]]).astype(np.float32)
    y = (q[:, 2].astype(np.float32) / max_dist).reshape(-1, 1)
    return X, y

X_train, y_train = make_data(train_q)
X_test,  y_test  = make_data(test_q)

class CrossEncoder(nn.Module):
    def __init__(self, d_in, mode='l1', r=R, s=S, emb_dim=EMB_DIM):
        super().__init__()
        self.mode, self.r, self.s, self.emb_dim = mode, r, s, emb_dim
        self.out_dim = emb_dim * 2
        self.net = nn.Sequential(
            nn.Linear(d_in, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HIDDEN, self.out_dim),
        )
    def forward(self, x):
        out = self.net(x)
        y_o, y_d = out[:, :self.out_dim//2], out[:, self.out_dim//2:]
        if self.mode == 'l1':
            return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True)
        else:
            sym = torch.abs(y_d[:, :self.r] - y_o[:, :self.r]).sum(dim=1, keepdim=True)
            asym = (y_d[:, self.r:self.r+self.s] - y_o[:, self.r:self.r+self.s]).sum(dim=1, keepdim=True)
            return sym + asym

def train(mode, Xt, yt, Xv, yv):
    m = CrossEncoder(Xt.shape[1], mode=mode).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt)), batch_size=4096, shuffle=True)
    Xvt, yvt = torch.from_numpy(Xv).cuda(), torch.from_numpy(yv).cuda()
    best = float('inf')
    for ep_i in range(EPOCHS):
        m.train()
        for bx, by in dl: bx, by = bx.cuda(), by.cuda(); opt.zero_grad(); nn.SmoothL1Loss()(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mre = (torch.abs(m(Xvt) - yvt) / (yvt + 1e-8)).mean().item()
            sch.step(mre)
            if mre < best: best = mre
    return best * 100

print(f"\n{'='*60}")
print(f"CE L1 vs L1Tilde ({CITY}, {args.model_class}, r={R}, s={S})")
print(f"{'='*60}")
for mode in ['l1', 'l1tilde']:
    t0 = time.time()
    mre = train(mode, X_train, y_train, X_test, y_test)
    print(f"  {mode:>8s}: Test MRE = {mre:.2f}%  ({time.time()-t0:.0f}s)")
