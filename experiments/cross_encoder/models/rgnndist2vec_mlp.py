"""
RGNNdist2vec + MLP Decoder — Cross-Encoder 改造。

架构变化:
    Original:  coord → GNN → emb_u, emb_v → L1/L1Tilde(emb_u, emb_v)
    MLP:       coord → GNN → emb_u, emb_v → concat → MLP → y_o, y_d → L1/L1Tilde(y_o, y_d)

MLP 的 concat 顺序固定 ([u_feat, v_feat])，权重矩阵的列位置创造了天然的不对称表示。
L1Tilde 在此基础上放大方向性。

支持 --use_l1tilde 切换 L1 / L1Tilde Decoder。
"""

import time
import numpy as np

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import k_hop_subgraph


class RGNNdist2vecMLP(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, layer_type,
                 node_attributes, edge_attributes, max_distance=1.0,
                 disable_edge_weight=True, directed=False,
                 use_l1tilde=False, l1tilde_r=62, l1tilde_s=2):
        super().__init__()
        self.max_distance = max_distance
        self.directed = directed
        self.use_l1tilde = use_l1tilde
        self.l1tilde_r = l1tilde_r
        self.l1tilde_s = l1tilde_s

        if use_l1tilde:
            assert l1tilde_r + l1tilde_s == n_hidden_2, \
                f"r({l1tilde_r})+s({l1tilde_s}) must == embed_dim({n_hidden_2})"

        # GNN layers (same as original)
        self.layer_type = layer_type
        if layer_type.lower() == "gcn":
            self.layer1 = GCNConv(n_input, n_hidden_1, add_self_loops=True,
                                  cached=False)
            self.layer2 = GCNConv(n_hidden_1, n_hidden_2, add_self_loops=True,
                                  cached=False)
        elif layer_type.lower() == "gat":
            self.layer1 = GATConv(n_input, n_hidden_1, add_self_loops=True,
                                  fill_value='mean')
            self.layer2 = GATConv(n_hidden_1, n_hidden_2, add_self_loops=True,
                                  fill_value='mean')
        elif layer_type.lower() == "sage":
            self.layer1 = SAGEConv(n_input, n_hidden_1)
            self.layer2 = SAGEConv(n_hidden_1, n_hidden_2)

        self.leaky_relu = nn.LeakyReLU()
        self.num_layers = 2

        # Build geometric data
        self.geometric_data = self.build_geometric_data(
            node_attributes, edge_attributes, layer_type,
            disable_edge_weight, directed)
        self.register_buffer('node_features', self.geometric_data.x)
        self.register_buffer('edge_index', self.geometric_data.edge_index)
        self.register_buffer('edge_weight', self.geometric_data.edge_weight)
        self.cached_embeddings = None

        # MLP Decoder: concat(u,v) -> y_o, y_d (improved: BatchNorm+Dropout)
        d_in = n_hidden_2 * 2
        hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_in),
        )

        dec = "L1Tilde" if use_l1tilde else "L1"
        print(f"RGNNdist2vecMLP: layer={layer_type}, decoder={dec}, "
              f"r={l1tilde_r}, s={l1tilde_s}" if use_l1tilde
              else f"RGNNdist2vecMLP: layer={layer_type}, decoder={dec}")

    def build_geometric_data(self, node_attributes, edge_attributes,
                             layer_type, disable_edge_weight, directed=False):
        node_features = torch.from_numpy(node_attributes).float()
        node_features = ((node_features - node_features.mean(dim=0))
                         / node_features.std(dim=0))

        edge_index = torch.from_numpy(
            edge_attributes[:, :2]).long().t().contiguous()
        edge_weight = torch.from_numpy(edge_attributes[:, 2]).float()
        if disable_edge_weight or layer_type.lower() == 'sage':
            edge_weight = None

        geometric_data = Data(x=node_features, edge_index=edge_index,
                              edge_weight=edge_weight)
        if not directed:
            geometric_data = ToUndirected()(geometric_data)
        return geometric_data

    def encode(self, node_features, edge_index, edge_weight=None):
        x = self.leaky_relu(self.layer1(node_features, edge_index,
                                        edge_weight))
        x = self.leaky_relu(self.layer2(x, edge_index, edge_weight))
        return x

    def forward(self, x1, x2, embeddings=None):
        if embeddings is None:
            if self.cached_embeddings is not None:
                embeddings = self.cached_embeddings
            else:
                self.cached_embeddings = self.encode(
                    self.geometric_data.x, self.geometric_data.edge_index,
                    self.geometric_data.edge_weight).detach().clone()
                embeddings = self.cached_embeddings

        emb1 = embeddings[x1]
        emb2 = embeddings[x2]

        # MLP Decoder: concat → y_o, y_d
        cat = torch.cat([emb1, emb2], dim=1)
        out = self.mlp(cat)
        y_o, y_d = torch.chunk(out, 2, dim=1)

        if self.use_l1tilde:
            r, s = self.l1tilde_r, self.l1tilde_s
            sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(
                dim=1, keepdim=True)
            asym = (y_d[:, r:r + s] - y_o[:, r:r + s]).sum(
                dim=1, keepdim=True)
            distances = sym + asym
        else:
            distances = torch.norm(y_d - y_o, p=1, dim=1, keepdim=True)

        return distances * self.max_distance

    def subgraph_extraction(self, x1, x2, geometric_data,
                            subgraph_node_map, num_layers):
        batch_nodes = torch.cat([x1, x2]).unique()
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = \
            k_hop_subgraph(
                node_idx=batch_nodes, num_hops=num_layers,
                edge_index=geometric_data.edge_index,
                num_nodes=geometric_data.num_nodes, relabel_nodes=True)
        subgraph_node_map[subgraph_nodes] = torch.arange(
            len(subgraph_nodes), device=subgraph_nodes.device)
        x1_sub = subgraph_node_map[x1]
        x2_sub = subgraph_node_map[x2]
        subgraph_features = geometric_data.x[subgraph_nodes]
        subgraph_edge_weight = None
        if geometric_data.edge_weight is not None:
            subgraph_edge_weight = geometric_data.edge_weight[edge_mask]
        return ((subgraph_features, subgraph_edge_index,
                 subgraph_edge_weight), x1_sub, x2_sub)

    def _train_step(self, geometric_data, x1, x2, y, criterion, optimizer,
                    num_layers, subgraph_node_map):
        optimizer.zero_grad()
        subgraph, x1_sub, x2_sub = self.subgraph_extraction(
            x1, x2, geometric_data, subgraph_node_map, num_layers)
        sf, sei, sew = subgraph
        embeddings = self.encode(sf, sei, sew)
        y_pred = self.forward(x1_sub, x2_sub, embeddings)
        y_pred = y_pred / self.max_distance
        y = y / self.max_distance
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, dataloader, criterion, optimizer,
            val_dataloader=None, epochs=1, display_step=10,
            device="cpu", fast_dev_run=False, time_limit=None, **kwargs):
        self.train()
        self.to(device)
        criterion.to(device)
        self.geometric_data.to(device)

        loss_epoch_history, loss_iter_history = [], []
        val_mre_epoch_history, time_history = [], []
        display_step = max(1, len(dataloader) // display_step)
        subgraph_node_map = torch.full(
            (self.geometric_data.num_nodes,), -1, dtype=torch.long,
            device=device)
        start_time = time.perf_counter()

        for epoch in range(epochs):
            running_loss = 0.0
            for idx, batch in enumerate(dataloader):
                i, j, d_ij = batch[0], batch[1], batch[2]
                d_ij = d_ij.unsqueeze(-1)
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
                loss = self._train_step(
                    self.geometric_data, i, j, d_ij, criterion,
                    optimizer, self.num_layers, subgraph_node_map)
                running_loss += loss.item()
                loss_iter_history.append(loss.item())

                if (idx + 1) % display_step == 0:
                    ls = (f"{loss.item():.8f}" if loss.item() <= 1.0
                          else f"{loss.item():.2f}")
                    print(f"Epoch: {epoch + 1:>2}/{epochs}, "
                          f"Batch: {idx + 1:>4} ({len(d_ij):>4}), "
                          f"Loss: {ls:>12}")

                if fast_dev_run:
                    break

            val_str = ""
            if val_dataloader is not None:
                val_preds, val_targets, _ = self.evaluate(
                    val_dataloader, device=device, verbose=False,
                    profile_time=False)
                val_mre = float(
                    np.mean(np.abs(val_preds - val_targets)
                            / np.maximum(val_targets, 1e-6)))
                val_mre_epoch_history.append(val_mre)
                val_str = f", Val MRE: {val_mre:.2%}"
                self.train()

            avg_loss = running_loss / len(dataloader)
            loss_epoch_history.append(avg_loss)
            elapsed = (time.perf_counter() - start_time) / 60
            time_history.append(elapsed)

            als = f"{avg_loss:.8f}" if avg_loss <= 1.0 else f"{avg_loss:.2f}"
            print(f"Epoch: {epoch + 1:>2}/{epochs}, "
                  f"Time: {elapsed:.1f}min, "
                  f"Avg Loss: {als:>12}{val_str}")

            if time_limit and elapsed >= time_limit:
                break

        return {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }

    def evaluate(self, dataloader, verbose=True, profile_time=True,
                 device="cpu", **kwargs):
        self.eval()
        self.to(device)
        self.geometric_data.to(device)

        predictions, targets = [], []
        total_time = 0.0

        with torch.no_grad():
            start = time.perf_counter()
            embeddings = self.encode(
                self.geometric_data.x, self.geometric_data.edge_index,
                self.geometric_data.edge_weight)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if verbose:
                et = ((time.perf_counter() - start)
                      / self.geometric_data.num_nodes)
                print(f"Embedding time per sample: "
                      f"{et * 1_000_000:.3f} us")

            for batch in dataloader:
                i, j, d_ij = batch[0], batch[1], batch[2]
                targets.append(d_ij.cpu().numpy())
                if profile_time:
                    start = time.perf_counter()
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j, embeddings)
                    outputs = outputs.cpu().numpy().ravel()
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    total_time += time.perf_counter() - start
                else:
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j, embeddings)
                    outputs = outputs.cpu().numpy().ravel()
                predictions.append(outputs)

        predictions = np.hstack(predictions)
        targets = np.hstack(targets)
        query_latency = total_time / len(targets)
        return predictions, targets, query_latency
