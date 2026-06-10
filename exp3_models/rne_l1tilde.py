"""
RNE + L̃₁ with Direction-Aware Training (策略三).

在解耦投影头 (策略一) 基础上，加入方向感知辅助 loss:
    main_loss:  MSE on d̂(u→v)
    aux_sym:    MSE on sym_dist vs (d_uv + d_vu)/2
    aux_asym:   MSE on asym_dist vs d_uv - d_vu

这样对称维度明确学习"基础距离"，非对称维度明确学习"方向偏差"，
解耦训练 (Decoupled Training) 大幅加速非对称维度的收敛。

References:
    [1] RNE (VLDB Journal 2022)
    [2] L̃₁ metric (VLDB 2025)
    [3] 解耦投影头 (实验1)
"""

import time
import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


class RNEL1Tilde(BaseModel):
    def __init__(self, num_nodes, embed_size, max_distance=1.0, parts=None,
                 l1tilde_r=62, l1tilde_s=2, aux_lambda=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s
        self.max_distance = max_distance
        self.aux_lambda = aux_lambda
        self.parts = torch.from_numpy(parts) if parts is not None else None
        assert l1tilde_r + l1tilde_s <= embed_size

        self.embedding = nn.Embedding(num_nodes, embed_size)
        nn.init.uniform_(self.embedding.weight, -3/2, 3/2)

        # 解耦投影头
        self.proj_sym = nn.Linear(embed_size, l1tilde_r)
        self.proj_asym = nn.Linear(embed_size, l1tilde_s)
        nn.init.normal_(self.proj_asym.weight, std=0.01)
        nn.init.zeros_(self.proj_asym.bias)

    def forward(self, x1, x2):
        emb1, emb2 = self.embedding(x1), self.embedding(x2)
        u_sym = self.proj_sym(emb1)
        u_asym = self.proj_asym(emb1)
        v_sym = self.proj_sym(emb2)
        v_asym = self.proj_asym(emb2)

        sym = torch.abs(v_sym - u_sym).mean(dim=1, keepdim=True)
        asym = (v_asym - u_asym).mean(dim=1, keepdim=True)
        x = sym + asym
        x = torch.clamp(x, min=0.0)
        return x * self.max_distance

    def _train_step_direction_aware(self, x1, x2, d_uv, d_vu, criterion,
                                     optimizer):
        """方向感知训练: 同时使用 d_uv 和 d_vu 计算辅助 loss。"""
        optimizer.zero_grad()

        emb1, emb2 = self.embedding(x1), self.embedding(x2)
        u_sym = self.proj_sym(emb1)
        u_asym = self.proj_asym(emb1)
        v_sym = self.proj_sym(emb2)
        v_asym = self.proj_asym(emb2)

        sym = torch.abs(v_sym - u_sym).mean(dim=1, keepdim=True)
        asym = (v_asym - u_asym).mean(dim=1, keepdim=True)
        pred = torch.clamp(sym + asym, min=0.0)

        # Normalize
        pred = pred / self.max_distance
        d_uv = d_uv / self.max_distance
        d_vu = d_vu / self.max_distance

        # Main loss: MSE on d(u→v)
        main_loss = criterion(pred, d_uv)

        # Auxiliary losses: 解耦训练
        # 对称维度 → 基础距离 (d_uv + d_vu) / 2
        base_dist = (d_uv + d_vu) / 2
        aux_sym = criterion(sym / self.max_distance, base_dist)
        # 非对称维度 → 方向偏差 d_uv - d_vu
        dir_diff = d_uv - d_vu
        aux_asym = criterion(asym / self.max_distance, dir_diff)

        loss = main_loss + self.aux_lambda * (aux_sym + aux_asym)
        loss.backward()
        optimizer.step()
        return loss

    def _train_step(self, x1, x2, y, criterion, optimizer):
        """标准训练 (无辅助 loss, 兼容原有 dataloader)。"""
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
            bidirectional=False, **kwargs):
        self.train()
        self.to(device)
        criterion.to(device)

        loss_epoch_history, loss_iter_history = [], []
        val_mre_epoch_history, time_history = [], []
        display_step = max(1, len(dataloader) // display_step)
        start_time = time.perf_counter()

        # 层次化训练
        if self.parts is not None:
            num_levels = self.parts.shape[1]
            h_epochs = [5]*(num_levels-1) + [10]
            self.parts = self.parts.to(device)
            prev_embeddings = None
            for level in range(num_levels):
                part_indices = self.parts[:, level]
                if level > 0:
                    with torch.no_grad():
                        prev_idx = self.parts[:, level-1]
                        self.embedding.weight.data[part_indices] = \
                            prev_embeddings[prev_idx]
                for epoch in range(h_epochs[level]):
                    for batch in dataloader:
                        if bidirectional and len(batch) >= 4:
                            i, j, d_uv, d_vu = batch
                            d_uv = d_uv.unsqueeze(-1).to(device)
                            d_vu = d_vu.unsqueeze(-1).to(device)
                            i = part_indices[i].to(device)
                            j = part_indices[j].to(device)
                            loss = self._train_step_direction_aware(
                                i, j, d_uv, d_vu, criterion, optimizer)
                        else:
                            i, j, d_ij = batch[:3]
                            d_ij = d_ij.unsqueeze(-1)
                            i = part_indices[i].to(device)
                            j = part_indices[j].to(device)
                            d_ij = d_ij.to(device)
                            loss = self._train_step(i, j, d_ij,
                                                     criterion, optimizer)
                prev_embeddings = self.embedding.weight.data.clone()

        # 标准/方向感知训练
        time_limit_checkpoint = None
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in dataloader:
                if bidirectional and len(batch) >= 4:
                    i, j, d_uv, d_vu = batch
                    d_uv = d_uv.unsqueeze(-1).to(device)
                    d_vu = d_vu.unsqueeze(-1).to(device)
                    i, j = i.to(device), j.to(device)
                    loss = self._train_step_direction_aware(
                        i, j, d_uv, d_vu, criterion, optimizer)
                else:
                    i, j, d_ij = batch[:3]
                    d_ij = d_ij.unsqueeze(-1)
                    i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
                    loss = self._train_step(i, j, d_ij, criterion, optimizer)

                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                if (idx := len(loss_iter_history)) % display_step == 0:
                    ls = f"{loss.item():.8f}" if loss.item() <= 1.0 \
                         else f"{loss.item():.2f}"
                    print(f"Epoch: {epoch+1:>2}/{epochs}, "
                          f"Batch: {(idx % len(dataloader))+1:>4}, "
                          f"Loss: {ls:>12}")

                if fast_dev_run:
                    break

            val_str = ""
            if val_dataloader is not None:
                val_preds, val_targets, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False,
                    profile_time=False)
                val_mre = float(np.mean(np.abs(val_preds - val_targets)
                                / np.maximum(val_targets, 1e-6)))
                val_mre_epoch_history.append(val_mre)
                val_str = f", Val MRE: {val_mre:.2%}"
                self.train()

            avg_loss = running_loss / len(dataloader)
            loss_epoch_history.append(avg_loss)
            elapsed = (time.perf_counter() - start_time) / 60
            time_history.append(elapsed)

            als = f"{avg_loss:.8f}" if avg_loss <= 1.0 else f"{avg_loss:.2f}"
            print(f"Epoch: {epoch+1:>2}/{epochs}, Time: {elapsed:.1f}min, "
                  f"Train Loss: {als:>12}{val_str}")

            if time_limit and elapsed >= time_limit \
               and time_limit_checkpoint is None:
                time_limit_checkpoint = {
                    "epoch": epoch + 1, "time_min": elapsed,
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
