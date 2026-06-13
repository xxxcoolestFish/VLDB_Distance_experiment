# Baseline Encoder-Decoder 架构全景分析

本项目共 17 个 baseline，按 Encoder-Decoder 架构分为 5 种类型。

---

## 概念定义

| 概念 | 含义 |
|------|------|
| **Encoder** | 将节点 ID → 向量的过程（GNN / 查表 / 冻结） |
| **Decoder** | 将两个向量 → 距离值的过程（数学公式 / MLP / GBDT） |
| **可索引** | Decoder 是纯数学公式，能建 M-Tree → O(log N) 查询 |
| **方向性** | d(u→v) ≠ d(v→u) 是否可能 |

---

## 一、类型总览

```
                         Encoder                            Decoder
                         ══════                            ══════
🔴 类型一   Embedding(N,64) 可学习                     对称公式 L1/cos/Lp
           ├── RNE, Path2Vec, ANEDA
           └── LpNorm (冻结坐标)

🟡 类型二   GNN(coord→64D) GCN/SAGE/GAT×2              对称公式 L1
           └── RGNN-SAGE, GAT, GCN

🟠 类型三   Embedding(N,64) 可学习                     Cross-Encoder MLP
           ├── NDist2Vec (4-branch)
           ├── VDist2Vec (3-layer)
           └── GeoDNN (坐标冻结)

🟣 类型四   冻结特征 坐标2D+landmark61D                  MLP/GBDT
           ├── EmbeddingNN, DistanceNN
           └── CatBoost, CatBoostNN

🔵 类型五   预计算最短路径矩阵                            min 聚合
           └── Landmark

🟢 我们的   Dist2Vec+GNN                                Pairwise MLP + L̃₁
           └── Dist2GNN
```

---

## 二、逐模型解剖

### 🔴 类型一：纯 Embedding + 对称 Decoder

无 Encoder 神经网络，每节点一行 64 维向量。Decoder 数学对称。**全部可索引，全部无方向性。**

| 模型 | Encoder | Decoder | 参数量 |
|------|------|------|:---:|
| **RNE** | `Embedding(N,64)` 随机 init→训练 | `mean(｜Δ｜) → clamp(0) × max` | N×64 |
| **Path2Vec** | `Embedding(N,64)` 随机 init→训练 | `1−cos_sim(emb_u, emb_v) × max/2` | N×64 |
| **ANEDA** | `Embedding(N,64)` 随机 init→训练 | 三选一: `1−cos` / `‖Δ‖_p` / `1+cos` | N×64 |
| **LpNorm** | 经纬度查表 冻结不可学 | `‖coord_u−coord_v‖_p` | **0** |

**L̃₁ 适用性**：
- RNE: ⭐⭐⭐ 理论最合适，但需解耦投影+方向感知（实验1+3 已证明有效）
- Path2Vec: ⭐⭐ 合适但 neighbor reg 与 L̃₁ 不兼容（实验0 失败）
- ANEDA: ⭐⭐ 可加 L̃₁ 为第四种 `distance_measure`
- LpNorm: ❌ 坐标不可学，维度仅 2，L̃₁ 完全不可用

---

### 🟡 类型二：GNN Encoder + L1 Decoder

2 层 GNN 将 2D 坐标编码为 64D embedding。图被 `ToUndirected()` 转无向。Decoder L1 数学对称。

| 模型 | Encoder 聚合方式 | Decoder | 方向性 |
|------|------|------|:---:|
| **SAGE** | `h_v = W₁h_v + W₂·mean(N(v))` | `‖GNN(u)−GNN(v)‖₁ × max` | ❌ |
| **GAT** | `h_v = Σ α_{vu}·Wh_u`, `α_{vu}≠α_{uv}` | `‖GNN(u)−GNN(v)‖₁ × max` | △ (Encoder 级) |
| **GCN** | `h_v = Σ h_u/√(d_v·d_u)` | `‖GNN(u)−GNN(v)‖₁ × max` | ❌ |

GAT 的 attention 权重不对称 → Encoder 隐式携带方向信息。但 Decoder L1 仍是数学对称的。

**MRE 基线** (Harbin): SAGE 6.51%, GAT 5.88%, GCN 32.02%

**L̃₁ 适用性**：
- SAGE: ⭐⭐ 方向能力最弱，L̃₁ 有改善空间（大图 -1.4~-1.9%）
- GAT: ⭐⭐ attention 提供隐式方向，L̃₁ 锦上添花（解耦投影 4.96%）
- GCN: ❌ 架构本身失效，换度量无意义

---

### 🟠 类型三：Embedding + MLP Cross-Encoder

Decoder 中 `cat(emb_u, emb_v)` → MLP。concat 的固定顺序 (`[u,v]` vs `[v,u]`) 赋予方向性。**精度高但不可索引。**

| 模型 | Encoder | Decoder | 参数量 |
|------|------|------|:---:|
| **NDist2Vec** | `Embedding(N,64)` 可学 | 4 分支 MLP 各负责一个距离尺度 | N×64 + 4×MLP |
| **VDist2Vec** | `Embedding(N,64)` 可学 | 3-layer MLP → sigmoid × max | N×64 + MLP |
| **GeoDNN** | 坐标查表 冻结 | 4-layer MLP(4→20→100→20→1) → sigmoid | 仅 MLP 参数 |

NDist2Vec 的核心设计：4 个 branch 各有独立 MLP + 可学习权重 v1~v4（初始化为 50/550/5500/max/2），输出 `sigmoid(branch) × vᵢ`。多尺度融合确保短距和长距都能精确预测。

**MRE 基线** (Harbin): NDist2Vec 11.28%, VDist2Vec 待测, GeoDNN 待测

**L̃₁ 适用性**：
- NDist2Vec: ❌ 4-branch MLP 太强，简单 Embedding+L̃₁ 无法替代（实验0: 62-72%）
- VDist2Vec: ⭐ MLP 更简单，有被 L̃₁ 替代的可能
- GeoDNN: ⭐ 需先加 Embedding 层

---

### 🟣 类型四：冻结特征 + MLP/GBDT

Embedding 是预计算的（坐标 2D + landmark 最短路径 61D），`freeze=True` 不更新。方向性来自 Decoder concat。

| 模型 | Encoder | Decoder |
|------|------|------|
| **EmbeddingNN** | 冻结坐标/预训练 | 聚合(had/sub/mean/cat) → 2-layer MLP |
| **DistanceNN** | 冻结坐标 | 聚合 → 3-layer MLP + Dropout + Softplus |
| **CatBoost** | 冻结 坐标+landmark | cat(260D) → GBDT 3000 树 |
| **CatBoostNN** | 冻结 坐标+landmark | cat(260D) → MLP(260→1024→512→1) |

EmbeddingNN/DistanceNN 的聚合方式决定方向性：`concat` 有方向性（位置固定），`subtract` 勉强（MLP(−Δ)≠−MLP(Δ)），`hadamard/mean` 无方向性。

CatBoost 特征向量（260D）：`[landmark_u(61), landmark_v(61), coord_u(2), coord_v(2), cos_sim(1), L1_dist(1)]`

**L̃₁ 适用性**：❌ 需加可学习 Embedding 层才能用 L̃₁。当前所有特征 freeze。

---

### 🔵 类型五：预计算（不学习）

| 模型 | Encoder | Decoder |
|------|------|------|
| **Landmark** | Dijkstra 预计算 N×k 距离矩阵 | `min_l(d(u,l) + d(v,l))` |

参数量为 **0**。方向性 ❌。L̃₁ 不适用。

---

### 🟢 我们的方法

| | Encoder | Decoder |
|------|------|------|
| **Dist2GNN** | Dist2Vec 预训练 Embedding + Landmark GNN | Pairwise MLP → y_o,y_d → **L̃₁(y_o,y_d)** |

唯一同时具备 GNN Encoder + MLP 交互 + L̃₁ 非对称度量的模型。方向性 ✅✅ 双保险，可索引 ✅（L̃₁ 是数学公式）。

---

## 三、方向性来源矩阵

```
                         Decoder
                   对称公式    MLP交互    L̃₁
                   ────────   ────────   ──
Encoder  无(查表)  RNE         NDist2Vec  RNE-L1Tilde ✅
         (可学)    Path2Vec    VDist2Vec  Path2Vec-L1Tilde*
                   ANEDA

Encoder  GNN       SAGE                    SAGE-L1Tilde ✅
                   GCN                     
                   GAT                     GAT-L1Tilde ✅

Encoder  冻结      LpNorm      EmbeddingNN
                           CatBoost/CatBoostNN
                           GeoDNN

预计算             Landmark
```

\* Path2Vec-L1Tilde 失败，因 neighbor reg 不兼容

---

## 四、可索引性矩阵

| 可索引 ✅ | 不可索引 ❌ |
|------|------|
| RNE, Path2Vec, ANEDA, LpNorm | NDist2Vec, VDist2Vec, GeoDNN |
| SAGE, GAT, GCN | EmbeddingNN(concat), DistanceNN(concat) |
| Landmark | CatBoost, CatBoostNN |
| **RNE-L1Tilde, SAGE-L1Tilde, GAT-L1Tilde** | |
| Dist2GNN | |

**L̃₁ 的核心价值**：将不可索引的"方向性 Decoder"替换为可索引版本。

---

## 五、L̃₁ 适用性排序

| 优先级 | 模型 | 理由 |
|:---:|------|------|
| 🔥 | **RNE** | 纯 Embedding 无方向性，L̃₁ 补充。解耦投影+方向感知已证明有效 (20.77%) |
| ⭐ | **SAGE** | GNN 结构先验 + L̃₁，大图有增益 |
| ⭐ | **GAT** | attention+L̃₁ 锦上添花 (4.96%) |
| ⭐ | **ANEDA** | 加 L̃₁ 为第四度量选项 |
| △ | **Path2Vec** | 需修复 neighbor reg 兼容性 |
| △ | **VDist2Vec** | MLP 简单，可能被 L̃₁ 替代 |
| ❌ | **NDist2Vec** | 4-branch MLP 太强无法替代 |
| ❌ | **CatBoost/CatBoostNN** | 需先加 Embedding 层 |
| ❌ | **LpNorm, Landmark, GeoDNN** | 特征不可学习 |
| ❌ | **GCN** | 架构已失效 |
