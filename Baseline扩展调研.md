# Baseline 扩展调研 — 新模型搜索与 L1Tilde 适用性分析

**日期**: 2026-06-17

**搜索范围**: 2023-2025 顶会顶刊 (VLDB, SIGMOD, ICDE, NeurIPS, ICML, ICLR, KDD, WWW, AAAI, IEEE TNNLS)，开源优先。研究领域：最短路径距离估计、图嵌入距离计算、边级预测、图 Transformer。

---

## 一、找到的候选 Baseline（共 14 个）

### A 类：直接做距离预测（最适合）

#### 1. GRED — Recurrent Distance Filtering (ICML 2024) ⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **发表** | ICML 2024 |
| **代码** | https://github.com/skeletondyh/GRED |

**思路**：按最短距离对邻居分层聚合，线性 RNN 编码逐跳表示。显式建模基于距离的邻域层次。

**能否应用 MLP+L1Tilde**：✅ Encoder 产生节点表示 → 加 MLP concat Decoder → L1Tilde

---

#### 2. PSGNN — Position-Sensing GNN (IEEE TNNLS 2024) ⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **发表** | IEEE TNNLS, March 2024 |
| **代码** | https://github.com/ZhenyueQin/PSGNN |

**思路**：可微分 anchor 选择 + 节点到 anchor 的距离 → 学习节点相对位置。pairwise 分类 +14% AUC，链接预测 +18% AUC。

**能否应用 MLP+L1Tilde**：✅ 已用 pairwise 距离特征 → MLP concat + L1Tilde 直接替代其分类 Decoder

---

#### 3. DLGNN — Distance-Enhanced GNN for Link Prediction (ICML 2021) ⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **发表** | ICML 2021（至今高引，SOTA） |
| **代码** | https://github.com/lbn187/DLGNN |

**思路**：随机选 anchor → 节点到 anchor 的最短路径距离均值作为节点间距离估计 → 距离向量与 GNN 边特征拼接。OGB 基准 SOTA。

**能否应用 MLP+L1Tilde**：✅ 已使用 anchor 距离 + MLP 组合 → CE+L1Tilde 自然扩展

---

#### 4. TGT — Triplet Graph Transformer (ICML 2024) ⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **发表** | ICML 2024 |
| **代码** | https://github.com/shamim-hussain/tgt |

**思路**：Edge-augmented Graph Transformer。第一阶段预测**原子间 pairwise 距离**，第二阶段用距离做分子属性预测。pairwise 通道允许两条共享节点的边交互。

**能否应用 MLP+L1Tilde**：✅ 已有 pairwise 距离预测模块 → 直接替换 Decoder 为 L1Tilde

---

### B 类：边级预测 / 链接预测（可适配）

#### 5. LPFormer — Adaptive Graph Transformer for Link Prediction (KDD 2024) ⭐⭐

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/HarryShomer/LPFormer |

**思路**：自适应学习每对候选链接的 pairwise encoding，用 attention 模块建模多种链接形成因素。

**能否应用 MLP+L1Tilde**：✅ 已有 pairwise encoding → 可替换 Decoder

---

#### 6. Edge Transformer (2024) ⭐⭐

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/luis-mueller/towards-principled-gts |

**思路**：在节点对上做全局 attention（而非节点上），不需要位置编码即达到 ≥3-WL 表达能力。直接建模 pairwise 关系。

**能否应用 MLP+L1Tilde**：✅ 天然处理 pairwise 关系 → L1Tilde 可直接替换其 readout

---

#### 7. SLATE — Supra-Laplacian Encoding for Dynamic Graphs (NeurIPS 2024) ⭐⭐

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/ykrmm/SLATE |

**思路**：跨注意力建模节点 pairwise 关系，显式边表示用于动态链接预测。9 个数据集上优于 MPNN+LSTM。

**能否应用 MLP+L1Tilde**：✅ 已有显式边表示 → 加 CE+L1Tilde

---

### C 类：图 Transformer / 位置编码（参考价值）

#### 8. HPLC — Hierarchical Position Embedding (WWW 2024) ⭐

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/kmswin1/HPLC |

**思路**：Landmark + 层次聚类 → 位置嵌入 → 链接预测。与 CatBoost 同源但更现代。

---

#### 9. SE-SGformer — Self-Explainable Graph Transformer (AAAI 2025) ⭐

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/liule66/SE-SGformer |

**思路**：符号随机游走位置编码 + k-近邻解码器，用于边的符号预测。

---

#### 10. UGT — Unified Graph Transformer (AAAI 2024) ⭐

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/NSLab-CUK/Unified-Graph-Transformer |

**思路**：编码节点间的结构距离和 k 步转移概率为自注意力偏置。

---

#### 11. Graphormer (NeurIPS 2021) — 最短路径距离作为位置编码

| 项目 | 详情 |
|------|------|
| **代码** | https://github.com/microsoft/Graphormer |

---

### D 类：非学习方法 / 不适用

| 模型 | 发表 | 原因 |
|------|------|------|
| G2H | VLDB 2024 | GPU 加速 hop-labeling 索引，非学习 |
| DHL | VLDB Journal 2025 | 时变道路网络索引，非学习 |
| Deep Distance Sensitivity Oracles | 2024 | 理论为主，无公开代码 |

---

## 二、筛选结果：可应用 MLP+L1Tilde 的模型

### 强推荐（A 类：距离预测原生领域）

| 优先级 | Baseline | 发表 | 开源 | 架构类型 | MLP+L1Tilde 适配 |
|:---:|------|:---:|:---:|------|------|
| 🔥 1 | **GRED** | ICML 2024 | ✅ | RNN距离聚合 | 加 CE Decoder |
| 🔥 2 | **PSGNN** | TNNLS 2024 | ✅ | 可学习Anchor GNN | 替换分类Dec |
| 🔥 3 | **DLGNN** | ICML 2021 | ✅ | Anchor距离GNN | 已有MLP+距离 |
| 🔥 4 | **TGT** | ICML 2024 | ✅ | 边增强Transformer | 已有pairwise模块 |

### 可考虑（B 类：边预测可适配）

| 优先级 | Baseline | 发表 | 开源 |
|:---:|------|:---:|:---:|
| ⭐ 5 | **LPFormer** | KDD 2024 | ✅ |
| ⭐ 6 | **Edge Transformer** | 2024 | ✅ |
| ⭐ 7 | **SLATE** | NeurIPS 2024 | ✅ |

## 三、原有 Baseline 对比

我们已有的 17 个 baseline 的发表年份分布：

| 年份 | 模型 |
|:---:|------|
| 2017-2019 | GeoDNN, Path2Vec, DistanceNN |
| 2020-2021 | VDist2Vec, CatBoost |
| 2022 | RNE, NDist2Vec |
| 2023 | EmbeddingNN, ANEDA |
| 2024 | RGCNdist2vec |
| 2025(our) | Dist2GNN |

新增的 **GRED (ICML 2024)**、**PSGNN (2024)**、**TGT (ICML 2024)** 可以填补 2024-2025 的时间缺口，且都来自顶会，有开源代码。

## 四、下一步建议

1. 下载 GRED 代码，适配到道路网络距离预测任务
2. 对 PSGNN 和 DLGNN 做同样的适配评估
3. 对适配后的模型运行三组实验：Bi-Encoder / CE+L1 / CE+L1Tilde
