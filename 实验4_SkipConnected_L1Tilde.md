# 实验4：Skip-Connected L̃₁ — 突破对称瓶颈

**日期**: 2026-06-11
**状态**: 计划中

---

## 一、问题：为什么 SAGE-L1Tilde 失败？

### 根因：方向性梯度被对称瓶颈阻断

```
SAGE-L1Tilde (当前, 失败):
    coord(2D) → GNN(mean) → h(64D) ─┬→ proj_sym(64→62) → sym
                                     └→ proj_asym(64→2) → asym
                                        ↑ 输入 h 已经被 mean() 对称化了！
```

SAGE 从输入到输出全是对称操作：

| 步骤 | 操作 | 特性 |
|:---:|------|------|
| 1 | `ToUndirected()` | 单边→双边，图结构对称化 |
| 2 | `mean(N(v))` | 邻居取均值，聚合对称化 |
| 3 | `‖h_u−h_v‖₁` | L1 距离，Decoder 对称化 |

`proj_asym` 试图从对称的输入 `h` 中提取方向信号 → 等同于从零中提取信号 → 输出纯噪声 → 干扰 `proj_sym` → 整体变差。

### 对比：为什么 RNE 和 GAT 成功？

| 模型 | 梯度路径 | 有对称瓶颈？ |
|------|------|:---:|
| **RNE** | Decoder → proj → **Embedding 直连** | ❌ |
| **GAT** | Decoder → proj → GNN(attention, α_AB≠α_BA) | ❌ 可绕过 |
| **SAGE** | Decoder → proj → GNN(mean) → ... | ✅ 被阻挡 |

---

## 二、方案：Skip-Connected L̃₁

### 核心思路

给非对称维度开一条**绕过 GNN 的直通路径**，方向性梯度不经过对称瓶颈。

```
Skip-Connected SAGE-L1Tilde:
                            ┌→ proj_sym(h) → sym      ← GNN 提供拓扑先验
    coord → GNN(mean) → h  ─┤
                            │
    Embedding_skip(N, s) ───┴→ proj_asym_skip(e) → asym  ← 直通！绕过 GNN
```

- `Embedding_skip(N, s)`: 一个独立的、小型的 Embedding 表，每节点 s 维
- `proj_asym_skip(s→s)`: 可选的线性投影
- 非对称维度的梯度 **不经过 GNN**，直接从 Decoder 反传到 Embedding_skip
- GNN 继续负责提供拓扑结构先验给对称维度

### 统一模板

```python
class SkipConnectedL1Tilde:
    def __init__(self, ...):
        # GNN Encoder (不变)
        self.gnn = ...  # SAGEConv / GATConv / GCNConv

        # 对称投影 (不变)
        self.proj_sym = nn.Linear(n_hidden_2, r)

        # 非对称跳过连接 — 新增
        self.embed_skip = nn.Embedding(num_nodes, s)
        nn.init.normal_(self.embed_skip.weight, std=0.01)  # 小权重初始化

    def forward(self, x1, x2):
        # GNN 前传 (给对称维度)
        h = self.gnn_encode()
        h1, h2 = h[x1], h[x2]
        u_sym = self.proj_sym(h1)
        v_sym = self.proj_sym(h2)
        sym = |v_sym − u_sym|.sum()

        # Skip 路径 (给非对称维度) — 绕过 GNN
        u_asym = self.embed_skip(x1)  # 直接查表
        v_asym = self.embed_skip(x2)
        asym = (v_asym − u_asym).sum()

        return (sym + asym) * max_distance
```

---

## 三、可覆盖的 Baseline

| 类型 | 模型 | 当前 L̃₁ 结果 | Skip-Connect 预期 | 关键 |
|------|------|:---:|:---:|------|
| GNN | **SAGE** | ❌ 7.00% | ✅ 预计改善 | asym 绕过 mean() |
| GNN | **GCN** | ❌ 30%+ | △ 可能改善 | 度归一化也是瓶颈 |
| GNN | **GAT** | ✅ 4.96% | ✅ 保持 | 可加 skip 但不必须 |
| 纯Embedding | **RNE** | ✅ 20.77% | ✅ 保持 | 本无瓶颈，skip 等价 |
| 纯Embedding | **Path2Vec** | ❌ 32-73% | ✅ 预计改善 | asym skip + 保留 neighbor reg |
| 纯Embedding | **ANEDA** | 未测 | ✅ 预计可行 | 加 skip L̃₁ 为新度量 |
| 纯Embedding | **VDist2Vec** | 未测 | ⭐ 有希望 | MLP 简单，skip 直出 |
| 纯Embedding | **NDist2Vec** | ❌ 62-72% | △ 仍难 | 4-branch MLP 太强 |
| 冻结特征 | **CatBoost等** | — | ❌ | 需先加 Embedding |
| 不可学 | **LpNorm, Landmark** | — | ❌ | 特征不可学习 |

**覆盖面**：从现在的 2 个（RNE、GAT）扩展到 **5-7 个**。

---

## 四、实验计划

### 4.1 验证实验：SAGE-Skip-L1Tilde

最关键的验证——SAGE 当前 7.00%（比 L1 的 6.51% 差）。如果 skip-connect 能修复到 <6.51%，方案成立。

```bash
# exp4_models/sage_skip_l1tilde.py
python train.py --model_class sage_skip_l1tilde \
    --gnn_layer sage --data_dir data/OSM_Harbin ...
```

### 4.2 推广实验

| 优先级 | 模型 | 理由 |
|:---:|------|------|
| 🔥 | **SAGE-Skip** | 最关键的验证 |
| ⭐ | **Path2Vec-Skip** | 保留 neighbor reg + asym skip |
| ⭐ | **ANEDA-Skip** | 新度量选项，改动最小 |
| △ | **GCN-Skip** | 尝试修复但期望不高 |
| △ | **VDist2Vec-Skip** | 简单 MLP 替代 |

### 4.3 对照

每个 Skip 模型同时对比：
1. 原版 L1
2. 硬切片 L̃₁
3. 解耦投影 L̃₁ (策略一)
4. Skip-Connected L̃₁（新）

---

## 五、预测

```
如果 SAGE-Skip-L1Tilde 成功 (< 6.51%):
    → Skip-Connect 被证明是突破对称瓶颈的有效机制
    → 可推广到 Path2Vec、ANEDA 等更多 baseline
    → Paper 贡献: "Skip-Connected L̃₁ bypasses symmetric bottlenecks in GNN encoders"

如果 SAGE-Skip-L1Tilde 失败 (≥ 7.00%):
    → SAGE 的问题不止是瓶颈，mean 聚合本身就不适合学习方向性
    → 需要更根本的架构改动（如有向 GNN）
```

---

## 六、文件计划

```
exp4_models/
├── __init__.py
├── sage_skip_l1tilde.py          # SAGE + skip-connected L̃₁
├── path2vec_skip_l1tilde.py      # Path2Vec + skip-connected L̃₁
└── aneda_skip_l1tilde.py         # ANEDA + skip-connected L̃₁
```

---

## 七、实验记录

| 日期 | 实验 | 模型 | Test MRE | 备注 |
|------|------|------|:---:|------|
| | | | | |
