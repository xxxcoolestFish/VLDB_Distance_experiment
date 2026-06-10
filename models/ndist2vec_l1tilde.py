"""
NDist2Vec + L̃₁ variant: 将 4-branch MLP Cross-Encoder 替换为解耦投影 L̃₁ Decoder。

References:
    [1] NDist2Vec (ISPRS 2022): Node with Landmark and New Distance to Vector Method
    [2] L̃₁ metric (VLDB 2025)

架构变化:
    Original:  emb_u, emb_v → cat → 4-branch MLP → v1*b1+v2*b2+v3*b3+v4*b4
    L̃₁:       emb_u, emb_v → proj_sym / proj_asym → L̃₁(emb_u, emb_v) * max_distance

这是 "用 L̃₁ 替代 MLP 交互" 的核心验证实验。
NDist2Vec 是 MLP 交互最强的 baseline（4-branch 多尺度），
如果 L̃₁ 能接近其精度，即证明 L̃₁ 能承担 MLP 的方向性角色，
同时保持 O(1) per-pair 计算和可索引性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class Ndist2vecL1Tilde(BaseModel):
    def __init__(self, num_nodes, embed_size=64, max_distance=1.0,
                 l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.max_distance = max_distance
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s

        assert l1tilde_r + l1tilde_s <= embed_size, \
            f"l1tilde_r({l1tilde_r}) + l1tilde_s({l1tilde_s}) must <= embed_size({embed_size})"

        # Embedding layer (same as original NDist2Vec)
        self.embedding = nn.Embedding(num_nodes, embed_size)
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.01)

        # 解耦投影头
        self.proj_sym = nn.Linear(embed_size, l1tilde_r)
        self.proj_asym = nn.Linear(embed_size, l1tilde_s)
        # 小权重初始化 → 训练初期等价纯对称 → 后期微调方向性
        nn.init.normal_(self.proj_asym.weight, std=0.01)
        nn.init.zeros_(self.proj_asym.bias)

        print(f"Ndist2vecL1Tilde: nodes={num_nodes}, embed={embed_size}, "
              f"r={l1tilde_r}, s={l1tilde_s}, max_distance={max_distance}")

    def forward(self, x1, x2):
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)

        # 解耦投影
        u_sym = self.proj_sym(emb1)
        u_asym = self.proj_asym(emb1)
        v_sym = self.proj_sym(emb2)
        v_asym = self.proj_asym(emb2)

        # L̃₁ 距离
        sym_dist = torch.abs(v_sym - u_sym).sum(dim=1, keepdim=True)
        asym_dist = (v_asym - u_asym).sum(dim=1, keepdim=True)

        # ReLU 保证非负，乘以 max_distance 匹配原始距离量级
        distance = F.relu(sym_dist + asym_dist) * self.max_distance

        return distance

    # 使用 BaseModel._train_step（不做归一化），与原版 NDist2Vec 一致
