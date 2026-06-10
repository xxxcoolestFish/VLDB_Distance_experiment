"""
Path2Vec + L̃₁ variant: 将对称的 cosine-similarity Decoder 替换为解耦投影 L̃₁。

References:
    [1] Path2Vec (ACL 2019): Making Fast Graph-based Algorithms with Graph Metric Embeddings
    [2] L̃₁ metric (VLDB 2025): Learning-Based Shortest Path Distance Estimation

架构变化:
    Original:  emb_u, emb_v → 1−cos_sim(emb_u, emb_v)
    L̃₁:       emb_u, emb_v → proj_sym / proj_asym → L̃₁(emb_u, emb_v)

保留原版的 neighbour regularization 和 L1 embedding regularization。
"""

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class Path2vecL1Tilde(BaseModel):
    def __init__(self, G, embed_size=64, max_distance=1.0,
                 regularize=True, l1factor=1e-10,
                 use_neighbors=True, neighbor_count=1, alpha=0.01,
                 l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.max_distance = max_distance
        self.regularize = regularize
        self.l1factor = l1factor
        self.use_neighbors = use_neighbors
        self.neighbor_count = neighbor_count
        self.alpha = alpha
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s

        assert l1tilde_r + l1tilde_s <= embed_size, \
            f"l1tilde_r({l1tilde_r}) + l1tilde_s({l1tilde_s}) must <= embed_size({embed_size})"

        num_nodes = G.number_of_nodes()
        self.neighbor_map = {i: list(G.successors(i)) for i in G.nodes()}

        # Embedding layer (same as original Path2Vec)
        self.embedding = nn.Embedding(num_nodes, embed_size)

        # 解耦投影头
        self.proj_sym = nn.Linear(embed_size, l1tilde_r)
        self.proj_asym = nn.Linear(embed_size, l1tilde_s)
        # 小权重初始化 → 训练初期等价纯对称 → 后期微调方向性
        nn.init.normal_(self.proj_asym.weight, std=0.01)
        nn.init.zeros_(self.proj_asym.bias)

        print(f"Path2vecL1Tilde: embed={embed_size}, r={l1tilde_r}, s={l1tilde_s}")

    def forward(self, x1, x2):
        # Embedding
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

        # ReLU 保证非负（不用 softplus，避免常数偏移 ln(2)≈0.69
        # 在归一化距离 [0,1] 上造成巨大偏差）
        distance = F.relu(sym_dist + asym_dist)

        return distance

    def _train_step(self, x1, x2, y, criterion, optimizer, sampled_neighbors):
        optimizer.zero_grad()
        y_pred = self.forward(x1, x2)

        # Normalize
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance

        loss = criterion(y_pred, y)

        if self.use_neighbors:
            src_nodes = x1.cpu().numpy()
            dst_nodes = x2.cpu().numpy()

            src_neighbors = [sampled_neighbors[int(node)] for node in src_nodes]
            dst_neighbors = [sampled_neighbors[int(node)] for node in dst_nodes]

            nodes, neighbors = [], []
            for node_i, neighbors_i in zip(src_nodes, src_neighbors):
                nodes.extend([node_i] * len(neighbors_i))
                neighbors.extend(neighbors_i)
            for node_j, neighbors_j in zip(dst_nodes, dst_neighbors):
                nodes.extend([node_j] * len(neighbors_j))
                neighbors.extend(neighbors_j)

            if len(nodes) > 0:
                nodes = torch.tensor(nodes, device=x1.device)
                neighbors = torch.tensor(neighbors, device=x1.device)

                # Neighbors should have high similarity → use dotproduct-style
                # (same as original Path2Vec)
                emb_n = self.embedding(nodes)
                emb_nb = self.embedding(neighbors)
                # 对邻居用 L̃₁ — 邻居距离应该很小
                ns_sym = self.proj_sym(emb_n)
                ns_asym = self.proj_asym(emb_n)
                nb_sym = self.proj_sym(emb_nb)
                nb_asym = self.proj_asym(emb_nb)
                neighbor_sym = torch.abs(nb_sym - ns_sym).sum(dim=1, keepdim=True)
                neighbor_asym = (nb_asym - ns_asym).sum(dim=1, keepdim=True)
                neighbor_dist = F.relu(neighbor_sym + neighbor_asym)

                neighbor_loss = torch.sum(neighbor_dist) / len(neighbor_dist)
                loss = (1 - self.alpha) * loss + self.alpha * neighbor_loss

        if self.regularize:
            l1_reg = torch.norm(self.embedding.weight, p=1)
            loss = loss + self.l1factor * l1_reg

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
        time_limit_checkpoint = None

        for epoch in range(epochs):
            running_loss = 0.0
            sampled_neighbors = {
                node: np.random.choice(neighbors,
                                       min(len(neighbors), self.neighbor_count),
                                       replace=False).tolist()
                for node, neighbors in self.neighbor_map.items()
            }

            for idx, batch in enumerate(dataloader):
                i, j, d_ij = batch
                d_ij = d_ij.unsqueeze(-1)
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)

                loss = self._train_step(i, j, d_ij, criterion, optimizer,
                                        sampled_neighbors)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                if (idx + 1) % display_step == 0:
                    loss_str = f"{loss.item():.8f}" if loss.item() <= 1.0 \
                               else f"{loss.item():.2f}"
                    print(f"Epoch: {epoch+1:>2}/{epochs}, "
                          f"Batch: {idx+1:>4} ({len(d_ij):>4} samples), "
                          f"Train Loss: {loss_str:>12}")

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

            avg_loss_str = f"{avg_loss:.8f}" if avg_loss <= 1.0 \
                           else f"{avg_loss:.2f}"
            print(f"Epoch: {epoch+1:>2}/{epochs}, "
                  f"Time elapsed/remaining/total: "
                  f"{time_elapsed:.2f}/{time_elapsed/(epoch+1)*(epochs-epoch-1):.2f}/"
                  f"{(time_elapsed + time_elapsed/(epoch+1)*(epochs-epoch-1)):.2f} min, "
                  f"Train Loss: {avg_loss_str:>12}{val_str}")

            if time_limit is not None and time_elapsed >= time_limit \
               and time_limit_checkpoint is None:
                time_limit_checkpoint = {
                    "epoch": epoch + 1,
                    "time_min": time_elapsed,
                    "train_loss": avg_loss,
                    "val_mre": val_mre_epoch_history[-1]
                    if val_mre_epoch_history else None,
                }
                print(f"[5min checkpoint] epoch={epoch+1}, "
                      f"train_loss={avg_loss:.6f}, "
                      f"val_mre={time_limit_checkpoint.get('val_mre')}")

        result = {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }
        if time_limit_checkpoint is not None:
            result["time_limit_checkpoint"] = time_limit_checkpoint
        return result
