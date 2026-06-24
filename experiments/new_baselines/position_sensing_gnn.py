"""
Position-Sensing GNN (PSGNN-style) 适配道路网络距离估计.

核心思路 (来自 PSGNN, TNNLS 2024):
    可学习 anchor 权重 + 距离加权特征 → pairwise距离预测

与 DLGNN-style 的关键区别:
    DLGNN: 随机 anchor, 固定 anchor 距离特征
    PSGNN: 可学习 anchor 权重, 每个 anchor 的贡献由模型自己决定

架构:
    1. 选 K 个 anchor (random/degree)
    2. 每个 anchor 有可学习权重 w_k → 加权 anchor 距离特征
    3. GNN 编码 → concat(gnn_u, weighted_anchor_u, gnn_v, weighted_anchor_v)
    4. MLP Decoder → L1 或 L1Tilde
"""
import time, numpy as np, torch, torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from models.basemodel import BaseModel
from utils.data_utils import select_landmarks, compute_landmark_distances


class PositionSensingGNN(BaseModel):
    """PSGNN-style: learnable anchor weights + GNN → pairwise距离预测"""

    def __init__(self, graph, num_nodes, node_attributes, edge_attributes,
                 layer_type='sage', max_distance=1.0, disable_edge_weight=True,
                 num_anchors=16,
                 use_l1tilde=False, l1tilde_r=2, l1tilde_s=62):
        super().__init__()
        self.max_distance = max_distance
        self.use_l1tilde = use_l1tilde
        self.l1tilde_r, self.l1tilde_s = l1tilde_r, l1tilde_s
        self.layer_type = layer_type
        self.num_anchors = num_anchors

        # ---- Learnable Anchor Weights (PSGNN核心) ----
        if num_anchors > 0:
            landmarks = select_landmarks(graph, num_anchors, strategy='random', seed=42)
            anchor_feat = compute_landmark_distances(graph, landmarks)
            anchor_feat = anchor_feat / (anchor_feat.max(axis=0, keepdims=True) + 1e-8)
            self.register_buffer('anchor_feat', torch.from_numpy(anchor_feat).float())
            # ★ 可学习 anchor 权重 (PSGNN-style)
            self.anchor_weight = nn.Parameter(torch.ones(num_anchors) * 0.1)
        else:
            self.register_buffer('anchor_feat', None)
            self.anchor_weight = None

        # ---- GNN Encoder ----
        n_input = node_attributes.shape[1]
        n_hidden_1, n_hidden_2 = 512, 64

        if layer_type == 'gcn':
            self.layer1 = GCNConv(n_input, n_hidden_1, add_self_loops=True, cached=False)
            self.layer2 = GCNConv(n_hidden_1, n_hidden_2, add_self_loops=True, cached=False)
        elif layer_type == 'gat':
            self.layer1 = GATConv(n_input, n_hidden_1, add_self_loops=True, fill_value='mean')
            self.layer2 = GATConv(n_hidden_1, n_hidden_2, add_self_loops=True, fill_value='mean')
        else:
            self.layer1 = SAGEConv(n_input, n_hidden_1)
            self.layer2 = SAGEConv(n_hidden_1, n_hidden_2)

        self.leaky_relu = nn.LeakyReLU()

        nf = torch.from_numpy(node_attributes).float()
        nf = (nf - nf.mean(0)) / nf.std(0)
        ei = torch.from_numpy(edge_attributes[:, :2]).long().t().contiguous()
        ew = torch.from_numpy(edge_attributes[:, 2]).float() if not disable_edge_weight else None
        if layer_type == 'sage': ew = None
        geo = Data(x=nf, edge_index=ei, edge_weight=ew)
        geo = ToUndirected()(geo)
        self.register_buffer('node_features', geo.x)
        self.register_buffer('edge_index', geo.edge_index)
        self.register_buffer('edge_weight', geo.edge_weight)
        self.cached_embeddings = None

        # ---- Decoder ----
        d_anchor = num_anchors if num_anchors > 0 else 0
        d_total = (n_hidden_2 + d_anchor) * 2
        hidden = 512
        self.decoder = nn.Sequential(
            nn.Linear(d_total, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_hidden_2 * 2),
        )

    def encode(self):
        if self.cached_embeddings is not None:
            return self.cached_embeddings
        x = self.leaky_relu(self.layer1(self.node_features, self.edge_index, self.edge_weight))
        x = self.leaky_relu(self.layer2(x, self.edge_index, self.edge_weight))
        self.cached_embeddings = x.detach().clone()
        return self.cached_embeddings

    def forward(self, x1, x2):
        gnn = self.encode()
        emb1, emb2 = gnn[x1], gnn[x2]

        if self.anchor_feat is not None:
            # ★ PSGNN关键: 可学习 anchor 权重加权
            w = torch.softmax(self.anchor_weight, dim=0)  # normalize
            a1 = self.anchor_feat[x1] * w.unsqueeze(0)
            a2 = self.anchor_feat[x2] * w.unsqueeze(0)
            cat_feat = torch.cat([emb1, a1, emb2, a2], dim=1)
        else:
            cat_feat = torch.cat([emb1, emb2], dim=1)

        out = self.decoder(cat_feat)
        y_o, y_d = out[:, :64], out[:, 64:]

        if self.use_l1tilde:
            r, s = self.l1tilde_r, self.l1tilde_s
            sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(dim=1, keepdim=True)
            asym = (y_d[:, r:r+s] - y_o[:, r:r+s]).sum(dim=1, keepdim=True)
            return (sym + asym) * self.max_distance
        else:
            return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True) * self.max_distance

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()
        yp = self.forward(x1, x2) / self.max_distance
        loss = criterion(yp, y / self.max_distance)
        loss.backward(); optimizer.step(); return loss

    def fit(self, dataloader, criterion, optimizer, val_dataloader=None,
            epochs=1, display_step=10, device="cpu", fast_dev_run=False,
            time_limit=None, **kwargs):
        self.train(); self.to(device); criterion.to(device)
        loss_ep, loss_it, val_mre, time_h = [], [], [], []
        dp = max(1, len(dataloader) // display_step); st = time.perf_counter()
        for epoch in range(epochs):
            rl = 0.0
            for batch in dataloader:
                i, j, d = batch[0], batch[1], batch[2].unsqueeze(-1)
                i, j, d = i.to(device), j.to(device), d.to(device)
                loss = self._train_step(i, j, d, criterion, optimizer)
                rl += loss.item(); loss_it.append(loss.item())
            avg = rl / len(dataloader); loss_ep.append(avg)
            el = (time.perf_counter() - st) / 60; time_h.append(el)
            vs = ""
            if val_dataloader:
                vp, vt, _ = self.evaluate(val_dataloader, device=device, verbose=False, profile_time=False)
                vm = float(np.mean(np.abs(vp-vt)/np.maximum(vt,1e-6)))
                val_mre.append(vm); vs = f", Val MRE: {vm:.2%}"; self.train()
            print(f"Epoch: {epoch+1:>2}/{epochs}, Time: {el:.1f}min, Loss: {avg:.6f}{vs}")
            if fast_dev_run: break
        return {"loss_epoch_history": loss_ep, "loss_iter_history": loss_it,
                "val_mre_epoch_history": val_mre, "time_history": time_h}
