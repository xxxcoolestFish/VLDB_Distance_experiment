"""
Node2Vec Baseline — KDD 2016 (Grover & Leskovec).

Node2Vec: Scalable Feature Learning for Networks.
使用有偏随机游走 + skip-gram 学习节点嵌入，纯无监督方法。

实验设计（遵循统一协议）:
  实验1 (原版):  Node2Vec 嵌入 → L1 距离 (无 MLP，无距离训练)
  实验2 (+MLP):  Node2Vec 嵌入 (冻结) → concat → MLP → y_o,y_d → L1
  实验3 (+L1Tilde): Node2Vec 嵌入 (冻结) → concat → MLP → y_o,y_d → L1Tilde(r=2,s=62)

嵌入生成: 在 __init__ 中如果未提供预计算嵌入，则自动调用 self.generate_embeddings()
用 networkx 做有偏随机游走 + gensim Word2Vec (skip-gram).
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn

from models.basemodel import BaseModel


# ── Node2Vec 嵌入生成（networkx + gensim）─────────────────────────────
def generate_node2vec_embeddings(G, embed_size=64, walk_length=40,
                                  num_walks=20, p=1.0, q=1.0,
                                  window=5, workers=4, seed=42,
                                  verbose=True):
    """
    对图 G (networkx DiGraph) 运行 Node2Vec 并返回 (num_nodes, embed_size) 的嵌入矩阵.

    参数:
        G: networkx 图（有向或无向）
        embed_size: 嵌入维度
        walk_length: 每条游走的长度
        num_walks: 每节点游走次数
        p, q: Node2Vec 的 return / in-out 参数 (p=q=1 退化为 DeepWalk)
        window: skip-gram 窗口大小
        workers: gensim 并行线程数
        seed: 随机种子
        verbose: 是否打印进度

    返回:
        embeddings: np.ndarray, shape (num_nodes, embed_size)
    """
    from gensim.models import Word2Vec

    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())

    if verbose:
        print(f"Node2Vec: num_nodes={num_nodes}, dim={embed_size}, "
              f"walks={num_walks}, walk_len={walk_length}, p={p}, q={q}")

    # 1. 预计算 alias 转移概率
    def get_alias_edge(src, dst):
        """对有向图，转移概率基于边权重."""
        neighbors = list(G.successors(dst))   # 从 dst 出发的邻居
        if len(neighbors) == 0:
            return None, None
        weights = []
        for nxt in neighbors:
            edge_data = G.get_edge_data(dst, nxt)
            w = float(edge_data.get('weight', 1.0)) if edge_data else 1.0
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        return neighbors, probs

    def alias_draw(neighbors, probs):
        """按概率采样"""
        r = np.random.rand() * sum(probs)
        cum = 0.0
        for nbr, prob in zip(neighbors, probs):
            cum += prob
            if r <= cum:
                return nbr
        return neighbors[-1]

    # 2. 生成随机游走
    walks = []
    np.random.seed(seed)
    for walk_idx in range(num_walks):
        if verbose and walk_idx % 5 == 0:
            print(f"  Random walks: {walk_idx}/{num_walks}")
        # 每轮游走随机排列节点顺序
        perm = np.random.permutation(nodes)
        for start_node in perm:
            walk = [start_node]
            current = start_node
            for _ in range(walk_length - 1):
                succ = list(G.successors(current))
                if len(succ) == 0:
                    break
                # Node2Vec 有偏游走: 用 p, q 调节回退/前进概率
                if len(walk) == 1:
                    # 第一步: 均匀随机选邻居
                    next_node = succ[np.random.randint(0, len(succ))]
                else:
                    prev = walk[-2]
                    # 对每个 successor 分配非归一化概率
                    unnorm_probs = []
                    for nxt in succ:
                        if nxt == prev:
                            unnorm_probs.append(1.0 / p)        # 回退
                        elif G.has_edge(nxt, prev) or G.has_edge(prev, nxt):
                            unnorm_probs.append(1.0)             # 同距离
                        else:
                            unnorm_probs.append(1.0 / q)         # 前进
                    total = sum(unnorm_probs)
                    probs = [up / total for up in unnorm_probs]
                    next_node = alias_draw(succ, probs)
                walk.append(next_node)
                current = next_node
            walks.append([str(n) for n in walk])

    if verbose:
        print(f"  Generated {len(walks)} walks, training skip-gram...")

    # 3. Word2Vec skip-gram
    model_w2v = Word2Vec(walks, vector_size=embed_size, window=window,
                         min_count=1, sg=1, workers=workers, seed=seed,
                         epochs=5)

    # 4. 提取嵌入矩阵
    embeddings = np.zeros((num_nodes, embed_size), dtype=np.float32)
    for i in nodes:
        embeddings[i] = model_w2v.wv[str(i)]

    if verbose:
        print(f"  Embeddings shape: {embeddings.shape}")

    return embeddings


# ── Node2Vec Baseline 模型 ─────────────────────────────────────────────
class Node2VecBaseline(BaseModel):
    """
    Node2Vec Baseline.

    mode='pure':      实验1 — 冻结嵌入 + L1 距离 (无 MLP)
    mode='mlp_l1':    实验2 — 冻结嵌入 + MLP + L1
    mode='mlp_l1tilde': 实验3 — 冻结嵌入 + MLP + L1Tilde
    """

    def __init__(self, num_nodes, embed_size=64, max_distance=1.0,
                 mode='pure', l1tilde_r=2, l1tilde_s=62,
                 init_embeddings=None, graph=None):
        """
        参数:
            num_nodes: 节点数
            embed_size: 嵌入维度
            max_distance: 最大距离 (用于缩放)
            mode: 'pure' | 'mlp_l1' | 'mlp_l1tilde'
            l1tilde_r, l1tilde_s: L1Tilde 的对称/非对称维度
            init_embeddings: 预计算的 Node2Vec 嵌入 (np.ndarray or None)
            graph: networkx 图 (用于生成 Node2Vec 嵌入)
        """
        super().__init__()
        self.max_distance = max_distance
        self.mode = mode
        self.embed_size = embed_size

        # 获取或生成 Node2Vec 嵌入
        if init_embeddings is not None:
            print(f"Loading precomputed Node2Vec embeddings, shape={init_embeddings.shape}")
        elif graph is not None:
            print("Generating Node2Vec embeddings from graph...")
            init_embeddings = generate_node2vec_embeddings(
                graph, embed_size=embed_size, verbose=True)
        else:
            raise ValueError("Must provide either init_embeddings or graph.")

        # Normalize and register
        init_embeddings = (init_embeddings - init_embeddings.mean(axis=0)) / \
                          (init_embeddings.std(axis=0) + 1e-8)
        init_embeddings = torch.from_numpy(init_embeddings).float()
        self.embedding = nn.Embedding.from_pretrained(init_embeddings, freeze=True)

        # 纯模式: 加一个可学习的 scale 参数 (使得优化器有东西训练)
        if mode == 'pure':
            self.scale = nn.Parameter(torch.tensor(1.0))

        # Build MLP if needed
        if mode.startswith('mlp'):
            d = embed_size * 2  # concat
            hidden = 512
            out_dim = embed_size * 2  # y_o, y_d
            self.mlp = nn.Sequential(
                nn.Linear(d, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, out_dim),
            )

        if mode == 'mlp_l1tilde':
            self.l1tilde_r = l1tilde_r
            self.l1tilde_s = l1tilde_s
            assert l1tilde_r + l1tilde_s == embed_size, \
                f"r+l must equal embed_size: {l1tilde_r}+{l1tilde_s} != {embed_size}"

        print(f"Node2VecBaseline: mode={mode}, embed_size={embed_size}")

    def forward(self, x1, x2):
        emb1 = self.embedding(x1)  # (B, D)
        emb2 = self.embedding(x2)  # (B, D)

        if self.mode == 'pure':
            # 实验1: L1(emb1, emb2) * scale 作为距离
            distances = torch.norm(emb1 - emb2, p=1, dim=1, keepdim=True)
            return distances * self.scale.abs() * self.max_distance

        # 实验2/3: concat → MLP → y_o, y_d → L1/L1Tilde
        cat = torch.cat([emb1, emb2], dim=1)
        out = self.mlp(cat)
        y_o, y_d = torch.chunk(out, 2, dim=1)

        if self.mode == 'mlp_l1tilde':
            r, s = self.l1tilde_r, self.l1tilde_s
            sym = torch.abs(y_d[:, :r] - y_o[:, :r]).sum(dim=1, keepdim=True)
            asym = (y_d[:, r:r + s] - y_o[:, r:r + s]).sum(dim=1, keepdim=True)
            distances = sym + asym
        else:  # mlp_l1
            distances = torch.norm(y_d - y_o, p=1, dim=1, keepdim=True)

        return distances * self.max_distance

    # ── 训练接口 ──────────────────────────────────────────────────────
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
            device="cpu", fast_dev_run=False, time_limit=None, **kwargs):
        self.train()
        self.to(device)
        criterion.to(device)

        loss_epoch_history, loss_iter_history = [], []
        val_mre_epoch_history, time_history = [], []
        display_step = max(1, len(dataloader) // display_step)
        start_time = time.perf_counter()

        for epoch in range(epochs):
            running_loss = 0.0
            for idx, batch in enumerate(dataloader):
                i, j, d_ij = batch[0], batch[1], batch[2]
                d_ij = d_ij.unsqueeze(-1)
                i, j, d_ij = i.to(device), j.to(device), d_ij.to(device)
                loss = self._train_step(i, j, d_ij, criterion, optimizer)
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

        predictions, targets = [], []
        total_time = 0.0

        with torch.no_grad():
            for batch in dataloader:
                i, j, d_ij = batch[0], batch[1], batch[2]
                targets.append(d_ij.cpu().numpy())
                if profile_time:
                    start = time.perf_counter()
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j)
                    outputs = outputs.cpu().numpy().ravel()
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    total_time += time.perf_counter() - start
                else:
                    i, j = i.to(device), j.to(device)
                    outputs = self.forward(i, j)
                    outputs = outputs.cpu().numpy().ravel()
                predictions.append(outputs)

        predictions = np.hstack(predictions)
        targets = np.hstack(targets)
        query_latency = total_time / len(targets) if profile_time else 0.0
        if verbose:
            mre = float(np.mean(np.abs(predictions - targets)
                               / np.maximum(targets, 1e-6)))
            print(f"Node2Vec({self.mode}) MRE: {mre:.2%}")
        return predictions, targets, query_latency
