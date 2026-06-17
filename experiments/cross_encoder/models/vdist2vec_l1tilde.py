"""
VDist2Vec + L1Tilde: MLP输出改为 y_o, y_d → L̃₁ Decoder。

原版: emb_u, emb_v → cat → MLP → sigmoid → d (标量)
L1Tilde: emb_u, emb_v → cat → MLP → y_o(64D), y_d(64D) → L̃₁(y_o, y_d)

MLP的concat顺序固定([u_feat, v_feat])，天然创造不对称表示。
L1Tilde在此基础上放大方向性。
"""

import torch
import torch.nn as nn

from models.basemodel import BaseModel


class Vdist2vecL1Tilde(BaseModel):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output,
                 max_distance=1.0, l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.max_distance = max_distance
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s
        assert l1tilde_r + l1tilde_s == n_hidden_1, \
            f"r({l1tilde_r})+s({l1tilde_s}) must == embed_dim({n_hidden_1})"

        # Embedding (same as original)
        self.embedding = nn.Embedding(n_input, n_hidden_1)
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.01)

        # MLP Encoder (same structure, changed output dim)
        self.fc1 = nn.Linear(n_hidden_1 * 2, n_hidden_2)
        self.fc2 = nn.Linear(n_hidden_2, n_hidden_3)
        # 输出改为 embed_dim*2 (y_o + y_d), 而不是 1
        self.fc3 = nn.Linear(n_hidden_3, n_hidden_1 * 2)

        # 初始化
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.trunc_normal_(layer.bias, mean=0.0, std=0.01)

        print(f"Vdist2vecL1Tilde: embed={n_hidden_1}, "
              f"mlp={n_hidden_2}→{n_hidden_3}→{n_hidden_1*2}, "
              f"r={l1tilde_r}, s={l1tilde_s}")

    def forward(self, x1, x2):
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)
        x = torch.cat((emb1, emb2), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)  # (B, embed_dim*2)

        y_o, y_d = torch.chunk(out, 2, dim=1)  # 各 (B, embed_dim)

        r, s = self.l1tilde_r, self.l1tilde_s
        sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(dim=1, keepdim=True)
        asym = (y_d[:, r:r + s] - y_o[:, r:r + s]).sum(dim=1, keepdim=True)

        return (sym + asym) * self.max_distance

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()
        y_pred = self.forward(x1, x2)
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss
