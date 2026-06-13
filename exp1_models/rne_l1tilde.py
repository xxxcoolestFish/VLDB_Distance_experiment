"""
RNE + L̃₁ with Disentangled Projection Heads.

策略一核心修改: 将原版硬切片改为独立解耦投影层。

原版 (models/rne_l1tilde.py):
    emb → diff = emb2 - emb1
    sym = mean(|diff[:62]|), asym = mean(diff[62:64])
    问题: 同一层输出的不同切片，梯度特性冲突 → 大图爆炸

改进版 (exp1_models/rne_l1tilde.py):
    emb → proj_sym(emb) → 62D, proj_asym(emb) → 2D
    sym = mean(|h_sym_v - h_sym_u|), asym = mean(h_asym_v - h_asym_u)
    proj_asym 小权重初始化 → 训练初期等价纯对称 → 后期微调方向性

References:
    [1] RNE (VLDB Journal 2022)
    [2] L̃₁ metric (VLDB 2025)
"""

import time
import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


class RNEL1Tilde(BaseModel):
    def __init__(self, num_nodes, embed_size, max_distance=1.0, parts=None,
                 l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.embed_size = embed_size
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s
        self.max_distance = max_distance
        self.parts = torch.from_numpy(parts) if parts is not None else None
        assert l1tilde_r + l1tilde_s <= embed_size, \
            f"r({l1tilde_r})+s({l1tilde_s}) must <= embed_size({embed_size})"
        print(f"RNEL1Tilde (disentangled): nodes={num_nodes}, "
              f"embed={embed_size}, r={l1tilde_r}, s={l1tilde_s}")

        # Embedding layer
        self.embedding = nn.Embedding(num_nodes, embed_size)
        nn.init.uniform_(self.embedding.weight, -3/2, 3/2)

        # 解耦投影头 — 策略一核心
        self.proj_sym = nn.Linear(embed_size, l1tilde_r)
        self.proj_asym = nn.Linear(embed_size, l1tilde_s)
        # 小权重初始化 → 粗到细学习
        nn.init.normal_(self.proj_asym.weight, std=0.01)
        nn.init.zeros_(self.proj_asym.bias)

    def forward(self, x1, x2):
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)

        # 解耦投影
        u_sym = self.proj_sym(emb1)
        u_asym = self.proj_asym(emb1)
        v_sym = self.proj_sym(emb2)
        v_asym = self.proj_asym(emb2)

        # L̃₁ 距离 (用 mean 保持与 RNE 一致)
        sym = torch.abs(v_sym - u_sym).mean(dim=1, keepdim=True)
        asym = (v_asym - u_asym).mean(dim=1, keepdim=True)
        x = sym + asym
        x = torch.clamp(x, min=0.0)
        x = x * self.max_distance

        return x

    def _train_step(self, x1, x2, y, criterion, optimizer):
        optimizer.zero_grad()
        y_pred = self.forward(x1, x2)
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, dataloader, criterion, optimizer,
            val_dataloader=None, epochs=1, display_step=10,
            device="cpu", time_limit=None, fast_dev_run=False,
            **kwargs):
        self.train()
        self.to(device)
        criterion.to(device)

        loss_epoch_history = []
        loss_iter_history = []
        val_mre_epoch_history = []
        time_history = []

        display_step = max(1, len(dataloader) // display_step)
        start_time = time.perf_counter()

        # 层次化训练 (METIS parts)
        if self.parts is not None:
            num_levels = self.parts.shape[1]
            h_epochs = [5]*(num_levels-1) + [10]
            self.parts = self.parts.to(device)

            prev_embeddings = None
            for level in range(self.parts.shape[1]):
                part_indices = self.parts[:, level]
                if level > 0:
                    with torch.no_grad():
                        prev_part_indices = self.parts[:, level-1]
                        self.embedding.weight.data[part_indices] = \
                            prev_embeddings[prev_part_indices]

                for epoch in range(h_epochs[level]):
                    for idx, batch in enumerate(dataloader):
                        i, j, d_ij = batch
                        d_ij = d_ij.unsqueeze(-1)
                        i = part_indices[i].to(device)
                        j = part_indices[j].to(device)
                        d_ij = d_ij.to(device)
                        loss = self._train_step(i, j, d_ij, criterion, optimizer)

                        if (idx + 1) % display_step == 0:
                            ls = f"{loss.item():.8f}" if loss.item() <= 1.0 \
                                 else f"{loss.item():.2f}"
                            print(f"[Level {level}] Epoch: {epoch+1:>2}/{h_epochs[level]}, "
                                  f"Batch: {idx+1:>4}, Loss: {ls:>12}")

                prev_embeddings = self.embedding.weight.data.clone()

        # 标准训练
        time_limit_checkpoint = None
        for epoch in range(epochs):
            running_loss = 0.0
            for idx, batch in enumerate(dataloader):
                i, j, d_ij = batch
                d_ij = d_ij.unsqueeze(-1)
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
                loss = self._train_step(i, j, d_ij, criterion, optimizer)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                if (idx + 1) % display_step == 0:
                    ls = f"{loss.item():.8f}" if loss.item() <= 1.0 \
                         else f"{loss.item():.2f}"
                    print(f"Epoch: {epoch+1:>2}/{epochs}, "
                          f"Batch: {idx+1:>4}, Train Loss: {ls:>12}")

                if fast_dev_run:
                    break

            val_str = ""
            if val_dataloader is not None:
                val_preds, val_targets, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False, profile_time=False)
                val_mre = float(np.mean(np.abs(val_preds - val_targets)
                                / np.maximum(val_targets, 1e-6)))
                val_mre_epoch_history.append(val_mre)
                val_str = f", Val MRE: {val_mre:.2%}"
                self.train()

            avg_loss = running_loss / len(dataloader)
            loss_epoch_history.append(avg_loss)
            time_elapsed = (time.perf_counter() - start_time) / 60
            time_history.append(time_elapsed)

            als = f"{avg_loss:.8f}" if avg_loss <= 1.0 else f"{avg_loss:.2f}"
            print(f"Epoch: {epoch+1:>2}/{epochs}, "
                  f"Time: {time_elapsed:.1f}min, "
                  f"Train Loss: {als:>12}{val_str}")

            if time_limit and time_elapsed >= time_limit \
               and time_limit_checkpoint is None:
                time_limit_checkpoint = {
                    "epoch": epoch + 1, "time_min": time_elapsed,
                    "val_mre": val_mre_epoch_history[-1]
                    if val_mre_epoch_history else None,
                }

        result = {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }
        if time_limit_checkpoint is not None:
            result["time_limit_checkpoint"] = time_limit_checkpoint
        return result
