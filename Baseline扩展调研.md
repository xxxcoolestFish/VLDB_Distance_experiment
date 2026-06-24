# Baseline 扩展调研 — 新模型搜索与 L1Tilde 适用性分析

**日期**: 2026-06-17

---

## 一、搜索策略

搜索范围：2023-2025 年顶会顶刊（VLDB, SIGMOD, ICDE, NeurIPS, ICML, ICLR, KDD, WWW, AAAI），开源优先。

研究领域：最短路径距离估计、图嵌入距离计算、道路网络距离预测。

---

## 二、找到的潜在 Baseline

### 2.1 GRED — Recurrent Distance Filtering for Graph Representation Learning ⭐⭐⭐

| 项目 | 详情 |
|------|------|
| **发表** | 2024 (顶会) |
| **GitHub** | https://github.com/skeletondyh/GRED |
| **论文** | arxiv.org/abs/2312.01538 |

**核心思路**：按最短距离对邻居节点进行分层聚合，使用线性 RNN（state-space model）编码逐跳表示。不需要 positional encoding，显式建模基于距离的邻域层次。

**架构**：
```
节点特征 → 按距离分桶(h=1,2,3...) → RNN编码 → 图读出
                ↑ 需要预计算最短路径距离
```

**能否应用 MLP+L1Tilde**：✅ 可以
- GRED 依赖预计算的距离来构建距离分桶
- 可以用我们的 GNN Encoder 替代其 RNN Encoder
- 在 Decoder 端加入 MLP concat → L1Tilde 来增强方向性
- 改动量：中等（需要将 GRED 适配到道路网络场景）

---

### 2.2 HPLC — Hierarchical Position Embedding with Landmarks and Clustering ⭐⭐

| 项目 | 详情 |
|------|------|
| **发表** | WWW 2024 |
| **GitHub** | https://github.com/kmswin1/HPLC |
| **论文** | WWW '24 |

**核心思路**：选高度数 landmark 节点作为参考点，用**到 landmark 的距离**、landmark 间距离、层次聚类分组作为位置嵌入，用于链接预测。提供了 power-law 图上平均路径长度的理论边界。

**架构**：
```
节点 → 到k个landmark的最短距离 → 层次聚类 → 位置嵌入 → 链接预测
```

**能否应用 MLP+L1Tilde**：✅ 可以
- HPLC 已经使用 landmark 距离作为特征
- 与我们现有的 CatBoost/CatBoostNN 类似（都用 landmark 距离）
- 可以加 MLP concat → L1Tilde 替代其链接预测 Decoder
- 改动量：较小

---

### 2.3 Graphormer — 使用最短路径距离做位置编码 ⭐

| 项目 | 详情 |
|------|------|
| **发表** | NeurIPS 2021 |
| **GitHub** | https://github.com/microsoft/Graphormer |
| **论文** | "Do Transformers Really Perform Bad for Graph Representation?" |

**核心思路**：将节点间的**最短路径距离**作为相对位置编码，与度中心性一起输入 Transformer 做图表示学习。

**能否应用 MLP+L1Tilde**：❌ 不太适合
- Graphormer 用于图/节点分类和回归，不是 pairwise 距离预测
- 需要大幅修改架构才能输出 pairwise 距离
- 但它的**距离作为位置编码**的思想可以参考

---

### 2.4 GraphGPS — 模块化图 Transformer ⭐

| 项目 | 详情 |
|------|------|
| **发表** | NeurIPS 2022 |
| **GitHub** | https://github.com/rampasek/GraphGPS |
| **论文** | "Recipe for a General, Powerful, Scalable Graph Transformer" |

**核心思路**：模块化框架，支持多种位置/结构编码 + 局部消息传递 + 全局注意力。

**能否应用 MLP+L1Tilde**：❌ 不适合
- 通用图表示学习框架，不针对距离预测
- 需要大量适配工作

---

### 2.5 Exphormer — 稀疏图 Transformer ⭐

| 项目 | 详情 |
|------|------|
| **发表** | ICML 2023 |
| **GitHub** | https://github.com/hamed1375/exphormer |
| **论文** | "Exphormer: Sparse Transformers for Graphs" |

**能否应用 MLP+L1Tilde**：❌ 不适合
- 图分类/节点分类，非距离预测
- 稀疏注意力 + 虚拟全局节点，与我们的任务不匹配

---

### 2.6 G2H — CPU-GPU Hybrid Labelling for Shortest Distance (VLDB 2024)

| 项目 | 详情 |
|------|------|
| **发表** | PVLDB Vol.18 (2024) |
| **GitHub** | https://github.com/sauccjy/GPU-H2H |
| **论文** | "A CPU-GPU Hybrid Labelling Algorithm for Massive Shortest Distance Queries on Road Networks" |

**核心思路**：GPU 加速的 hop-labeling 索引，非学习方法。

**能否应用 MLP+L1Tilde**：❌
- 传统的 2-hop labeling 索引方法，无神经网络
- 无法应用 L1Tilde

---

### 2.7 DHL — Time-Dependent Indexing (VLDB Journal 2025)

| 项目 | 详情 |
|------|------|
| **发表** | VLDB Journal 2025 |

**能否应用 MLP+L1Tilde**：❌
- 时变道路网络的距离索引，非学习方法

---

## 三、筛选结果

| Baseline | 发表 | 开源 | 可应用MLP+L1Tilde | 推荐度 |
|------|:---:|:---:|:---:|:---:|
| **GRED** | 2024 顶会 | ✅ | ✅ | 🔥 推荐 |
| **HPLC** | WWW 2024 | ✅ | ✅ | ⭐ 可考虑 |
| Graphormer | NeurIPS 2021 | ✅ | ❌ | — |
| GraphGPS | NeurIPS 2022 | ✅ | ❌ | — |
| Exphormer | ICML 2023 | ✅ | ❌ | — |
| G2H | VLDB 2024 | ✅ | ❌ | 非学习方法 |
| DHL | VLDB J 2025 | — | ❌ | 非学习方法 |

---

## 四、最终推荐

### 确定新增的 Baseline

| 优先级 | Baseline | 理由 |
|:---:|------|------|
| 🔥 **1** | **GRED** | 直接使用最短路径距离做分层聚合，架构与我们的 GNN+Decoder 模式兼容，可加 MLP+L1Tilde。2024 年发表，顶会，开源。 |
| ⭐ **2** | **HPLC** | Landmark-based 方法，与我们的 CatBoost 同源但更现代（WWW 2024）。可直接在其 landmark 距离特征上叠加 CE+L1Tilde。 |

### 建议

GRED 和 HPLC 各代表一种方法范式：
- **GRED**：距离驱动的分层聚合 → 与我们实验 0-3 的 GNN/Embedding 类互补
- **HPLC**：landmark 距离特征 → 与 CatBoost 类互补

两个都可尝试加入 MLP+L1Tilde，预计改动量在工作量可控范围内。
