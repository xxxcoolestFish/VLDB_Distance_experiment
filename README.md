# VLDB Distance — L1Tilde Metric Ablation Study

Code and experiments for: *"Does L̃₁ asymmetric metric improve over L1 for directed road network distance estimation?"*

---

## 项目目录结构

```
VLDB_Distance_experiment/
├── train.py                          # 主训练入口 (支持所有 model_class)
├── models/                           # 原始 17 个 baseline 模型 (不变)
├── utils/                            # 共享工具 (data_utils, torch_utils 等)
├── scripts/                          # 数据准备、benchmark 脚本
│
├── experiments/                      # ★ 实验代码
│   └── cross_encoder/               # Cross-Encoder + L1Tilde 实验
│       ├── models/                   # CE 模型 (5个)
│       │   ├── rgnndist2vec_mlp.py   #   GNN → concat → MLP → L1/L1Tilde
│       │   ├── rne_mlp.py            #   RNE → concat → MLP → L1/L1Tilde
│       │   ├── aneda_mlp.py          #   ANEDA → concat → MLP → L1/L1Tilde
│       │   ├── ndist2vec_l1tilde.py  #   NDist2Vec + L1Tilde
│       │   └── vdist2vec_l1tilde.py  #   VDist2Vec + L1Tilde
│       └── scripts/                  # CE 运行脚本 (4个)
│           ├── cross_encoder_correct.py   # 正确方法: 冻结GNN→CE L1/L1Tilde
│           ├── cross_encoder_emb.py       # 冻结Embedding→CE L1/L1Tilde
│           ├── run_all_baselines_cd.sh    # Chengdu 批量运行
│           └── run_full_pipeline_harbin.sh # Harbin 完整 pipeline
│
├── ablation_study/                   # 导师的实验代码 (参考)
├── original/                         # 历史参考代码
├── modified/                         # 历史参考 L1Tilde 变体
│
├── 最终实验总结.md                    # 全部实验结果汇总
├── 实验结果深度分析.md                # 从架构角度的深度分析
├── CrossEncoder_L1Tilde_实验记录.md   # 实验过程记录
└── 第一次消融分析.md                  # 原始消融分析
```

---

## 实验方法

### Cross-Encoder + L1Tilde（正确方法）

遵循导师的两阶段训练策略：

```
阶段1: 训练 Bi-Encoder (GNN / Embedding) → 保存 checkpoint
阶段2: 冻结 Encoder → 提取特征 → Cross-Encoder → L1 / L1Tilde(r=2,s=62)

   concat(feat_u, feat_v) → MLP(BN+Dropout) → y_o(64D), y_d(64D)
                                                    ↓
                                          L1:   ‖y_d − y_o‖₁
                                          L1Tilde:  r=2 对称 + s=62 非对称
```

**关键参数**: GNN 30ep lr=0.001, CE 50ep lr=0.001, SmoothL1, random_500k queries

---

## 运行方式

### 1. 训练原有 Bi-Encoder Baseline

```bash
# 单 GNN baseline
python train.py --model_class rgnndist2vec --gnn_layer sage \
    --data_dir data/OSM_Chengdu --query_dir data/OSM_Chengdu/random_500k \
    --epochs 30 --device cuda --loss smoothl1 --learning_rate 0.001 --seed 42

# 支持的 model_class: 见下方 "All Supported Models"
```

### 2. Cross-Encoder 实验（正确方法）

```bash
# GNN-based: 使用 cross_encoder_correct.py
python experiments/cross_encoder/scripts/cross_encoder_correct.py \
    --data_dir data/OSM_Chengdu \
    --gnn_ckpt results/xxx/saved_models/rgnndist2vec_OSM_Chengdu_random_500k.pt \
    --gnn_layer sage --epochs 50 --r 2 --s 62

# Embedding-based: 使用 cross_encoder_emb.py
python experiments/cross_encoder/scripts/cross_encoder_emb.py \
    --data_dir data/OSM_Chengdu \
    --model_class rne --ckpt results/xxx/saved_models/rne_OSM_Chengdu_random_500k.pt \
    --epochs 50 --r 2 --s 62
```

### 3. 批量运行（完整 Pipeline）

```bash
# Chengdu: 所有 7 个 baseline × Cross-Encoder L1/L1Tilde
bash experiments/cross_encoder/scripts/run_all_baselines_cd.sh

# Harbin: SAGE + GAT 完整 pipeline（先训 GNN 再 CE）
bash experiments/cross_encoder/scripts/run_full_pipeline_harbin.sh
```

### 4. 新增的 model_class（用于端到端训练，不推荐）

```bash
# GNN + MLP Cross-Encoder (端到端，不如两阶段)
python train.py --model_class rgnndist2vec_mlp --gnn_layer sage ...
python train.py --model_class rgnndist2vec_mlp_l1tilde --gnn_layer sage --l1tilde_r 2 --l1tilde_s 62 ...

# 纯 Embedding + MLP Cross-Encoder
python train.py --model_class rne_mlp ...
python train.py --model_class rne_mlp_l1tilde --l1tilde_r 2 --l1tilde_s 62 ...
python train.py --model_class aneda_mlp ...
python train.py --model_class aneda_mlp_l1tilde --l1tilde_r 2 --l1tilde_s 62 ...

# L1Tilde 变体 (直接替换)
python train.py --model_class ndist2vec_l1tilde --l1tilde_r 2 --l1tilde_s 62 ...
python train.py --model_class vdist2vec_l1tilde --l1tilde_r 2 --l1tilde_s 62 ...
```

---

## 核心实验结果

### 三城对比（两阶段 Cross-Encoder, r=2,s=62）

| City | 节点 | 最佳模型 | Best MRE |
|------|:---:|------|:---:|
| **Chengdu** | 111K | GAT Bi-Encoder | **3.38%** |
| **Beijing** | 163K | GAT Bi-Encoder | **3.68%** |
| **Harbin** | 43K | SAGE Bi-Encoder | **5.10%** |

### L1Tilde 有效性（仅在路网有方向性的城市有效）

| Baseline | Chengdu | Beijing | Harbin |
|------|:---:|:---:|:---:|
| SAGE | **-1.17pp** ✅ | +0.17 | +1.67 |
| GAT | +0.71 | **-1.01pp** ✅ | +6.24 |
| RNE | **-3.44pp** ✅ | — | — |
| ANEDA | **-21.11pp** 🔥 | — | — |

L1Tilde 有效: 4/14 (29%)。仅在 Encoder 缺乏方向能力 + 路网方向性强的场景有效。

### 核心结论

1. **GAT Bi-Encoder 是一致最佳方案**（3.38-5.12%），无需任何改造
2. **r=2,s=62（2对称+62非对称）是正确配置**；r=62,s=2 导致 L1Tilde 崩溃
3. **两阶段训练 > 端到端**：GNN 和 MLP 必须分开训练
4. **Bi-Encoder 始终 >> Cross-Encoder**（差距 2-6x）
5. **Harbin 方向信号太弱**，所有 L1Tilde 改造均失败

---

## Data Preparation

### Data Sources

Road network data is downloaded from OpenStreetMap via OSMnx for 4 Chinese cities:

| City | Approx. Nodes | Approx. Edges |
|------|:-----------:|:-----------:|
| Harbin | 44K | 108K |
| Chengdu | 111K | 275K |
| Qingdao | 119K | 294K |
| Beijing | 163K | 402K |

### Quick Start

```bash
# 1. Install dependencies
pip install torch torch_geometric numpy pandas networkx scikit-learn tqdm matplotlib seaborn osmnx routingkit_cch catboost

# 2. Download and prepare data
python scripts/prepare_data.py

# 3. Run a single Bi-Encoder model
python train.py --model_class rgnndist2vec --gnn_layer gat \
    --data_dir data/OSM_Harbin --query_dir data/OSM_Harbin/random_500k \
    --epochs 30 --device cuda --force_shift 0 --log_dir results/test

# 4. Run Cross-Encoder experiment (correct method)
python experiments/cross_encoder/scripts/cross_encoder_correct.py \
    --data_dir data/OSM_Chengdu --gnn_ckpt <checkpoint_path> \
    --gnn_layer sage --epochs 50 --r 2 --s 62

# 5. Run full benchmark
bash scripts/run_full_benchmark.sh
```

---

## All Supported Models

```bash
# Non-ML baselines
python train.py --model_class landmark ...
python train.py --model_class lpnorm --p_norm 1 ...

# NN baselines
python train.py --model_class geodnn / ndist2vec / vdist2vec ...
python train.py --model_class distancenn / embeddingnn / catboostnn / catboost ...

# Functional baselines
python train.py --model_class path2vec / aneda / rne ...

# GNN baselines (L1 metric)
python train.py --model_class rgnndist2vec --gnn_layer gat/sage/gcn ...

# L1→L̃₁ variants
python train.py --model_class rgnndist2vec_l1tilde / rne_l1tilde / lpnorm_l1tilde ...

# Cross-Encoder variants (experiments/)
python train.py --model_class rgnndist2vec_mlp / rgnndist2vec_mlp_l1tilde ...
python train.py --model_class rne_mlp / rne_mlp_l1tilde / aneda_mlp / aneda_mlp_l1tilde ...
python train.py --model_class ndist2vec_l1tilde / vdist2vec_l1tilde ...

# Our method
python train.py --model_class dist2gnn ...
```
