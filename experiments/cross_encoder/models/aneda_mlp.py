"""ANEDA + MLP Decoder — 支持L1/L1Tilde。保留原版可学习Embedding。"""
import torch, torch.nn as nn
from models.basemodel import BaseModel

class ANEDAMLP(BaseModel):
    def __init__(self, num_nodes, embed_size, max_distance=1.0,
                 use_l1tilde=False, l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.max_distance, self.use_l1tilde = max_distance, use_l1tilde
        self.l1tilde_r, self.l1tilde_s = l1tilde_r, l1tilde_s
        if use_l1tilde: assert l1tilde_r + l1tilde_s <= embed_size

        self.embedding = nn.Embedding(num_nodes, embed_size)

        d_in = embed_size * 2; hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x1, x2):
        e1, e2 = self.embedding(x1), self.embedding(x2)
        out = self.mlp(torch.cat([e1, e2], dim=1))
        y_o, y_d = torch.chunk(out, 2, dim=1)
        if self.use_l1tilde:
            r, s = self.l1tilde_r, self.l1tilde_s
            sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(dim=1, keepdim=True)
            asym = (y_d[:, r:r + s] - y_o[:, r:r + s]).sum(dim=1, keepdim=True)
            return (sym + asym) * self.max_distance
        return torch.norm(y_d - y_o, p=1, dim=1, keepdim=True) * self.max_distance

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()
        yp = self.forward(x1, x2) / self.max_distance
        loss = criterion(yp, y / self.max_distance)
        loss.backward(); optimizer.step(); return loss
