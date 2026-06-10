# 实验0：给 Baseline 赋予方向性 —— 用 L̃₁ 替代 / 补充欠缺的方向能力

**日期**: 2026-06-10
**状态**: 计划中

---

## 一、核心思想

### 1.1 问题

当前 17 个 baseline（含 Dist2GNN）中，方向性的来源分为三类：

| 方向性来源 | 模型 | 可索引？ |
|------|------|:---:|
| **完全没有** — Decoder 数学对称 | LpNorm, Landmark, RNE, Path2Vec, ANEDA, RGNN-SAGE, RGNN-GCN | ✅ 可索引 |
| **Decoder 交互带来** — concat→MLP/GBDT 隐式学不对称 | GeoDNN, VDist2Vec, NDist2Vec, EmbeddingNN, DistanceNN, CatBoost, CatBoostNN | ❌ 不可索引 |
| **Attention 带来** — encoder 内嵌不对称，但 decoder 仍对称 | RGNN-GAT | ✅ 可索引 |

### 1.2 核心假设

> **L̃₁ 可以在不损失可索引性的前提下，为模型赋予方向能力。**

具体而言：

1. **类别 A（无方向性模型）**：纯 Embedding + 对称 Decoder 的模型（Path2Vec、ANEDA），将对称 Decoder 替换为 L̃₁，赋予其方向能力。如果 MRE 下降 → L̃₁ 确实"创造"了方向性。

2. **类别 B（MLP 交互带来方向性）**：这些模型的 MLP Decoder 虽然带来了方向性，但代价是**不可索引**（每次查询需要跑 MLP）。如果我们把 MLP 替换为 L̃₁，并让 Encoder 承担方向性的学习：

   - 如果 MRE 接近原版 → L̃₁ 用 O(1) 计算替代了 MLP，保持了方向性的同时恢复了可索引性
   - 如果 MRE 反而更好 → L̃₁ 的显式非对称维度比 MLP 的隐式学习更高效

### 1.3 Sci-Fi 级别的终极假设

如果实验成功，论文可以宣称：

> *"MLP-based decoders learn directionality implicitly through concatenation — but at the cost of breaking indexability. We show that L̃₁ can replace these MLP decoders entirely: providing equal or better directional modeling while preserving O(1) per-pair inference compatible with M-Tree spatial indexing."*

---

## 二、实验对象

### 2.1 全部 Baseline 及其架构解剖

每个模型的 `forward()` 流程、方向性来源、以及 L̃₁ 改造切入点。

---

#### (1) LpNorm

```
forward: coord_u, coord_v → ‖coord_u − coord_v‖_p
```

| 项 | 说明 |
|------|------|
| Embedding | ❌ 直接使用经纬度物理坐标（不可学习） |
| Encoder | ❌ 无 |
| Decoder | Lp 范数，**完全对称** |
| 方向性 | ❌ 无 |
| 可索引 | ✅ |

**L̃₁ 改造可行性**：❌ 不适用。物理坐标不可学习，且维度只有 2（r=1,s=1 无意义）。

---

#### (2) Landmark

```
forward: node_u, node_v → 查表 d(landmark, u)、d(landmark, v) → min(d1+d2)
```

| 项 | 说明 |
|------|------|
| Embedding | 预计算的最短路径距离查表（冻结） |
| Encoder | ❌ 无 |
| Decoder | `min_l(d(l, u) + d(l, v))`，**完全对称** |
| 方向性 | ❌ 无 |
| 可索引 | ✅ |

**L̃₁ 改造可行性**：❌ 不适用。Landmark 的距离矩阵是预计算的（非学习）。

---

#### (3) RNE ★

```
forward: emb_u, emb_v → mean(|emb_u − emb_v|) → * max_distance
```

| 项 | 说明 |
|------|------|
| Embedding | `nn.Embedding(N, 64)`, 随机初始化，**可学习** |
| Encoder | ❌ 无（就是查表） |
| Decoder | `mean(|Δ|)` 对称 + 硬截断 `clamp(min=0)` |
| 方向性 | ❌ 无 |
| 可索引 | ✅ |

**已有 L̃₁ 变体**：`rne_l1tilde` — 硬切片 62/2，**大图上爆炸** (+12~16% MRE)。

**本次改进方向**：在 `rne_l1tilde` 基础上加**解耦投影头** + softplus。

---

#### (4) Path2Vec ★

```
forward: emb_u, emb_v → 1 − cosine_similarity(emb_u, emb_v) → * max_distance/2
```

| 项 | 说明 |
|------|------|
| Embedding | `nn.Embedding(N, 64)`, **可学习** |
| Encoder | ❌ 无 |
| Decoder | `1−cos_sim`，**完全对称** |
| 方向性 | ❌ 无 |
| 额外 | `_train_step` 有邻居正则化 loss |
| 可索引 | ✅ |

**当前无 L̃₁ 变体**。适合创建 `path2vec_l1tilde.py`。

**L̃₁ 改造方案**：
- 保留：`nn.Embedding(N, 64)` + 邻居正则化 loss
- 替换：Decoder `1−cos_sim` → L̃₁(emb_u, emb_v)
- 保留原有 neighbor regularization 作为辅助 loss
- 可选：加解耦投影头

---

#### (5) ANEDA ★

```
forward: emb_u, emb_v → 依 distance_measure 选:
    "inv_dotproduct": 1−cos_sim  (对称)
    "norm":           ‖emb_u−emb_v‖_p  (对称)
    "dotproduct":     1+cos_sim  (对称)
```

| 项 | 说明 |
|------|------|
| Embedding | `nn.Embedding(N, 64)`, **可学习** |
| Encoder | ❌ 无 |
| Decoder | 三种可选，**全部对称** |
| 方向性 | ❌ 无 |
| 可索引 | ✅ |

**当前无 L̃₁ 变体**。适合创建 `aneda_l1tilde.py`。

**L̃₁ 改造方案**：
- 保留：可学习 Embedding
- 新增距离度量选项：`distance_measure="l1tilde"`，内部 `r=62, s=2`
- 可选加解耦投影头

---

#### (6) RGNN-SAGE ★

```
forward: coord → GNN(SAGEConv×2) → 64D emb → ‖emb_u − emb_v‖_1 → *max_distance
```

| 项 | 说明 |
|------|------|
| Embedding | GNN 输出（以经纬度为输入特征） |
| Encoder | 2-layer SAGEConv（mean 聚合），**无向图** |
| Decoder | L1 范数，**完全对称** |
| 方向性 | ❌ 无（图被 ToUndirected() 转无向 + mean 对称） |
| 可索引 | ✅ |

**已有 L̃₁ 变体**：`rgnndist2vec_l1tilde` — 硬切片 62/2。大图上有改善 (-1.4~-1.9%)。

**本次改进方向**：在现有 `rgnndist2vec_l1tilde` 基础上加**解耦投影头**。

---

#### (7) RGNN-GCN

同 SAGE 架构，GCNConv 替代 SAGEConv。已有 L̃₁ 变体，GCN 本身是瓶颈（MRE 20-32%），暂不优化。

---

#### (8) RGNN-GAT

```
forward: coord → GNN(GATConv×2) → 64D emb → ‖emb_u − emb_v‖_1 → *max_distance
```

| 项 | 说明 |
|------|------|
| Embedding | GNN 输出 |
| Encoder | 2-layer GATConv，attention 权重 α_{vu} **不对称** |
| Decoder | L1 范数，**完全对称** |
| 方向性 | △ Encoder 级弱方向性（embedding 质量被 attention 改善） |
| 可索引 | ✅ |

**已有 L̃₁ 变体**：`rgnndist2vec_l1tilde`。消融结果：与 L1 无差异（Δ 在噪声级）。

**本次改进方向**：加解耦投影头看能否有微弱增益。但 GAT 本身 L̃₁ 增益有限，优先级低。

---

#### (9) GeoDNN

```
forward: coord_u, coord_v → cat([coord_u, coord_v]) → MLP(4→20→100→20→1) → sigmoid * max_distance
```

| 项 | 说明 |
|------|------|
| Embedding | 物理坐标查表（冻结） |
| Encoder | ❌ 无（坐标直接 concat） |
| Decoder | **Cross-Encoder**: MLP 处理 concat(u,v) |
| 方向性 | ✅ Decoder 级（MLP(a,b) ≠ MLP(b,a)，concat 顺序固定） |
| 可索引 | ❌ 每次查询需跑 MLP |

**L̃₁ 改造方案**：
- 新增可学习 Embedding 层（替代坐标冻结查表）
- 替换 MLP Decoder → L̃₁ Decoder
- 相当于把 GeoDNN 变成"GeoDNN-L1Tilde"：embedding(u)、embedding(v) → L̃₁

---

#### (10) VDist2Vec ★

```
forward: emb_u, emb_v → cat([emb_u, emb_v]) → MLP(128→100→20→1) → sigmoid * max_distance
```

| 项 | 说明 |
|------|------|
| Embedding | `nn.Embedding(N, 64)`, **可学习** |
| Encoder | ❌ 无 |
| Decoder | **Cross-Encoder**: MLP(concat(u,v)) |
| 方向性 | ✅ Decoder 级 |
| 可索引 | ❌ |

**L̃₁ 改造方案**：
- 保留：可学习 Embedding
- 替换：MLP Decoder → L̃₁ Decoder（纯 Bi-Encoder）
- 核心假设验证：L̃₁ 能否达到 MLP 的精度？

---

#### (11) NDist2Vec ★

```
forward: emb_u, emb_v → cat([emb_u, emb_v]) → 4-branch MLP → sigmoid → v1*b1+v2*b2+v3*b3+v4*b4
```

| 项 | 说明 |
|------|------|
| Embedding | `nn.Embedding(N, 64)`, 可学习 |
| Encoder | ❌ 无 |
| Decoder | **Cross-Encoder**: 4 个独立 MLP branch，各负责一个距离尺度（v1 短距 ~50m, v4 长距 ~max/2） |
| 方向性 | ✅ Decoder 级 |
| 可索引 | ❌ |
| 特点 | 4 个可学习权重 v1~v4 用于多尺度融合 |

**L̃₁ 改造方案**：
- 保留：可学习 Embedding
- 替换：4-branch MLP → L̃₁ Decoder
- NDist2Vec 是 "MLP 交互" 方向性最强的 baseline（4-branch 专门处理多尺度），用它验证 L̃₁ 能否替代 MLP 最有说服力

---

#### (12) EmbeddingNN

```
forward: emb_u, emb_v → aggregate(hadamard/subtract/mean/concat) → MLP(embed_size→500→1) → sigmoid * max_distance
```

| 项 | 说明 |
|------|------|
| Embedding | 冻结（坐标或预训练 embedding） |
| Encoder | ❌ 无 |
| Decoder | 聚合 + MLP |
| 方向性 | concat: ✅ (位置固定); subtract: △ (MLP 不对称); mean/hadamard: ❌ |
| 可索引 | concat 不可索引; subtract 勉强; mean/hadamard 可索引 |

**L̃₁ 改造方案**：
- concat/subtract 变体 → 替换 MLP 为 L̃₁
- mean/hadamard 变体 → 加 L̃₁ 赋予方向性

---

#### (13) DistanceNN

同 EmbeddingNN 结构，MLP 更深（3 层 + Dropout + Softplus）。改造方案相同。

---

#### (14-15) CatBoost / CatBoostNN

```
forward: coord_u, coord_v, landmark_u, landmark_v → cat + cos_sim + eucl_dist → GBDT(3000 trees) / MLP
```

| 项 | 说明 |
|------|------|
| Embedding | 冻结（坐标 + landmark 距离） |
| Encoder | ❌ 无 |
| Decoder | GBDT/MLP(128*2+2+1+1=260 → 1024 → 512 → 1) |
| 方向性 | ✅ Decoder 级 |
| 可索引 | ❌ |

**L̃₁ 改造方案**：
- 新增可学习 Embedding 层（替掉冻结特征）
- 替换 GBDT/MLP → L̃₁

---

#### (16) Dist2GNN

已有 L̃₁ Decoder + Pairwise MLP，双保险。不作改动。

---

### 2.2 实验优先级矩阵

| 优先级 | 模型 | 类别 | 理由 |
|:---:|------|:---:|------|
| 🔥 **P0** | **NDist2Vec** | B | 最强的 MLP 交互基线，验证 "L̃₁ 替代 MLP" 最有说服力 |
| 🔥 **P0** | **Path2Vec** | A | 完全对称 Decoder + 可学习 Embedding，最简单的 A 类代表 |
| ⭐ **P1** | **RNE** | A | 已爆炸，加解耦投影修复 → 看能否收敛 |
| ⭐ **P1** | **VDist2Vec** | B | 类似 NDist2Vec 但更简单（单分支 MLP） |
| ⭐ **P1** | **SAGE-L1Tilde** | A(已存在) | 已有变体，加解耦投影观察改进幅度 |
| 🔹 **P2** | **ANEDA** | A | 多度量支持，可加 `l1tilde` 选项 |
| 🔹 **P2** | **GAT-L1Tilde** | 已有 | 加解耦投影看微弱增益 |
| 🔹 **P2** | **GeoDNN** | B | 需加 Embedding 层，改动较大 |
| 🔹 **P2** | **EmbeddingNN** | B | 聚合方式多，需分别处理 |

---

## 三、统一改造模板

所有 L̃₁ 改造遵循统一模板（根据文档 `code优化.md` 的策略一）：

### 3.1 解耦投影头 (Disentangled Projection Heads)

```
原始（硬切片）:
    emb = encoder(x)
    sym = |emb[:r] − emb[:r]|          # ← 同一层输出的不同切片
    asym = emb[r:r+s] − emb[r:r+s]    # ← 梯度特性冲突

改进（解耦投影）:
    h = encoder(x)                     # GNN/Embedding 输出通用特征
    h_sym = proj_sym(h)                # nn.Linear(hidden, r)  ← 独立参数
    h_asym = proj_asym(h)              # nn.Linear(hidden, s)  ← 小权重初始化
    sym_dist = |h_sym_u − h_sym_v|.sum()
    asym_dist = (h_asym_v − h_asym_u).sum()
    distance = softplus(sym_dist + asym_dist)  # 保证非负
```

### 3.2 权重初始化策略

```python
# proj_asym 用小权重 → 训练初期等价于纯对称模型 → 后期微调方向性
nn.init.normal_(self.proj_asym.weight, std=0.01)
nn.init.zeros_(self.proj_asym.bias)
```

### 3.3 对于不同架构的适配

**Embedding-only 模型** (RNE, Path2Vec, ANEDA, NDist2Vec, VDist2Vec):
```
原 emb(x) → L̃₁ 距离
改 emb(x) → proj_sym / proj_asym → L̃₁ 距离
```
保持简单：embedding 作为通用特征 h，投影头做方向性拆分。

**GNN 模型** (SAGE, GAT, GCN):
```
原 GNN(coord) → L̃₁ 距离
改 GNN(coord) → proj_sym / proj_asym → L̃₁ 距离
```
保持：GNN 是通用特征提取器。

---

## 四、评估协议

### 4.1 数据集

**仅使用 Harbin** (OSM_Harbin, 44K 节点, 108K 边)。快速迭代验证。

### 4.2 指标

| 指标 | 说明 |
|------|------|
| **Train/Test MRE** | 全局 Mean Relative Error |
| **High-Asym MRE** | 只评估 `d(u→v)` 与 `d(v→u)` 差异 >20% 的 query pair |
| **Asym Ratio Histogram** | 展示不同非对称程度下的 MRE 分布 |

### 4.3 对照实验

每个改造模型同时跑：
1. 原版（L1 / MLP Decoder）
2. L̃₁ 硬切片版本（当前实现）
3. L̃₁ 解耦投影版本（新实现）

对比三种版本在同一 Harbin 数据上的 MRE。

---

## 五、文件结构

```
models/
├── path2vec_l1tilde.py          # 新增: Path2Vec → L̃₁
├── aneda_l1tilde.py             # 新增: ANEDA → L̃₁
├── ndist2vec_l1tilde.py         # 新增: NDist2Vec → L̃₁
├── vdist2vec_l1tilde.py         # 新增: VDist2Vec → L̃₁
├── rne_l1tilde.py               # 修改: 加解耦投影头
├── rgnndist2vec_l1tilde.py      # 修改: 加解耦投影头
└── ...
```

`train.py` 中新增 4 个 `model_class` 分支：`path2vec_l1tilde`, `aneda_l1tilde`, `ndist2vec_l1tilde`, `vdist2vec_l1tilde`。

---

## 六、实验记录

### 第一轮 (2026-06-11) — 解耦投影头 + softplus

**配置**: Harbin 44K, 20 epoch, 解耦投影头 (proj_sym + proj_asym), softplus

| 实验 | 模型 | Decoder | Train MRE | Test MRE | 判定 |
|------|------|------|:---:|:---:|------|
| 1 | Path2Vec | 1−cos_sim (对称) | 4.91% | **4.95%** | baseline |
| 2 | Path2Vec-L1Tilde | L̃₁ (proj + softplus) | — | **782%** | ✗ softplus 偏移 0.69 致命 |
| 3 | NDist2Vec | 4-branch MLP | 8.72% | **11.28%** | baseline |
| 4 | NDist2Vec-L1Tilde | L̃₁ (proj + softplus) | — | **75.27%** (1ep) | 未跑完 |

**关键发现**：
- `softplus(0)=0.693` 对归一化距离 [0,1] 造成巨大偏移 → Path2Vec 爆炸
- NDist2Vec 用原始距离（43 万米），softplus 偏移相对较小但仍有影响

### 第二轮 (2026-06-11) — softplus→ReLU 修复

**修复**: softplus 改为 ReLU（避免常数偏移 ln(2)≈0.693）

| 实验 | 模型 | Train MRE | Test MRE | 判定 |
|------|------|:---:|:---:|------|
| 1 | Path2Vec 原版 | 4.91% | **4.95%** | 保持不变 |
| 2 | Path2Vec-L1Tilde | 31.81% | **32.73%** | △ 大幅改善但仍远差于原版 |
| 3 | NDist2Vec 原版 | 8.72% | **11.28%** | 保持不变 |
| 4 | NDist2Vec-L1Tilde | 64.60% | **72.12%** | ✗ 训练停滞，loss 0.015 但 MRE 72% |

**分析**：

**Path2Vec-L1Tilde (32.73% vs 4.95%)**：
- 原版 neighbor regularization 是为 cos_sim 设计的（loss 会变负值，邻居正则项 bounded [0,2]）
- L̃₁ 的 ReLU 输出无上界 → neighbor_loss 量级不对等 → alpha=0.5 时邻居项主导训练
- 修复方向：降低 alpha 或去掉邻居正则

**NDist2Vec-L1Tilde (72.12% vs 11.28%)**：
- `_train_step` 中 `y_pred / max_distance` 导致梯度被除以 43 万 → 学习几乎停滞
- 原版 NDist2Vec 有 sigmoid + 可学习 scale(v1~v4) 来匹配距离量级
- 修复方向：去掉 max_distance 归一化，让输出直接预测原始距离

### 第三轮 (2026-06-11) — 去掉归一化 / 去掉邻居正则

| 实验 | 模型 | Train MRE | Test MRE | 判定 |
|------|------|:---:|:---:|------|
| 2-r2 | Path2Vec-L1Tilde (no neighbor) | 40.11% | **42.90%** | ✗ 不降反升 |
| 4-r2 | NDist2Vec-L1Tilde (raw loss) | 60.57% | **62.86%** | ✗ Loss 剧烈震荡 |

**分析 — 为什么都失败了？**

**Path2Vec-L1Tilde**: 去掉邻居正则后更差（42.90% vs 32.73%）。
原版 Path2Vec 的邻居正则 (`alpha=0.5`) 实际上承担了**自监督结构先验**的角色——
强制邻居节点的 embedding 相似，为纯查表式 embedding 提供了图的局部拓扑信息。
这是 Path2Vec 能达到 4.95% 的关键，不是可选的辅助项。

**NDist2Vec-L1Tilde**: 直接预测原始距离导致 loss 在 2 亿 ~ 960 亿之间剧烈震荡。
ReLU 输出无上界，MSE 梯度随预测值线性增长 → 预测越大 → 梯度越大 → 
权重更新越剧烈 → 下个 batch 预测崩坏。形成恶性循环。

### 核心发现（重要！）

**三轮实验揭示了一个清晰的规律：**

```
L̃₁ + 纯 Embedding (无结构先验) = 失败
L̃₁ + GNN Encoder (SAGE) = 有增益 (-1.4~-1.9%)
L̃₁ + GNN Encoder + Attention (GAT) = 无增益 (attention 已经解决方向性)
MLP Cross-Encoder (NDist2Vec) = 精度最高但不可索引
```

L̃₁ 不是 "免费的午餐"——它需要一个**好的 Encoder** 来组织 embedding 空间。
纯查表式 embedding (RNE, Path2Vec, NDist2Vec-L1Tilde) 缺乏结构先验，
L̃₁ 的 62/2 维度分工无法有效学习。

这个发现本身是 paper 的重要 contribution：*L̃₁ 的适用条件是架构必须具备足够的结构编码能力。*

### 下一步建议

- **P0 继续**: Path2Vec-L1Tilde 保留邻居正则，修复其与 L̃₁ 的兼容性
- **P1 优先**: 先攻克 RNE-L1Tilde（解耦投影头）——RNE 是纯 embedding 中最简单的，修复它的爆炸问题
- **P1 优先**: SAGE-L1Tilde 加解耦投影头——已有正增益，看能否进一步改善
- NDist2Vec / VDist2Vec → L̃₁ 暂时搁置，MLP 交互太强无法替代 |
