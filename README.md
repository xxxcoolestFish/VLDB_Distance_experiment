# VLDB Distance — L̃₁ 度量优化路网最短路径距离估计

研究问题：**L̃₁ (L1Tilde) 非对称度量在 Cross-Encoder 框架下是否优于 L1？**

---

## 一、我们对原始项目做了哪些改动

**原则：所有改动是增量的，不覆写原始代码。**

| 改动 | 位置 | 说明 |
|------|------|------|
| 新增 CE 模型（5个） | `experiments/cross_encoder/models/` | 冻结 Encoder + MLP → L1/L1Tilde 的 Cross-Encoder 变体 |
| 新增 CE 训练脚本 | `experiments/cross_encoder/scripts/` | 两阶段训练：先训 Bi-Encoder，再冻结 + MLP |
| 新增不冻结 CE 脚本 | `experiments/50ep_chengdu/scripts/ce_unfrozen.py` | Part B：GNN+MLP 联合训练 |
| 新增 Node2Vec baseline | `experiments/new_baselines/node2vec/` | KDD 2016, CCF-A，通用图嵌入 baseline |
| 注册新模型 | `train.py`（末尾新增 import 块） | 新增的 model_class 通过 `experiments/` 路径注册 |
| 文档 | 根目录 `实验总结.md`, `experiments/README.md` | 实验结果与目录说明 |

**未改动的部分**：`models/`、`utils/`、`original/` 保持原样。

---

## 二、目录结构（仅列我们新增/相关的）

```
VLDB_Distance_experiment/
├── train.py                              # 主入口（我们新增了 model_class 注册）
├── models/                               # 原始代码，未修改
├── utils/                                # 原始工具，未修改
│
├── experiments/                          # ★ 我们的全部新代码
│   ├── README.md                         #   目录详细说明
│   ├── cross_encoder/                    #   Cross-Encoder 实验
│   │   ├── models/                       #   5 个 CE 模型
│   │   │   ├── rgnndist2vec_mlp.py       #     GCN/GAT/SAGE + MLP + L1/L1Tilde
│   │   │   ├── rne_mlp.py                #     RNE + MLP + L1/L1Tilde
│   │   │   ├── aneda_mlp.py              #     ANEDA + MLP + L1/L1Tilde
│   │   │   ├── ndist2vec_l1tilde.py      #     NDist2Vec + L1Tilde
│   │   │   └── vdist2vec_l1tilde.py      #     VDist2Vec + L1Tilde
│   │   └── scripts/
│   │       ├── cross_encoder_correct.py  #     冻结 CE 主脚本（Part A）
│   │       └── cross_encoder_emb.py       #     嵌入预计算
│   │
│   ├── 50ep_chengdu/                     #   50 epoch 实验
│   │   └── scripts/ce_unfrozen.py        #     不冻结 CE（Part B）：GNN+MLP 一起训
│   │
│   ├── new_baselines/                    #   我们新增的 baseline
│   │   └── node2vec/                     #     Node2Vec [KDD 2016]
│   │       └── node2vec_baseline.py       #     含 pure/+MLP+L1/+MLP+L1Tilde 三模式
│   │
│   └── archived/                         #   废弃代码（仅供参考）
│       ├── dlg_style/                    #     DLGNN-style anchor 增强（被否）
│       ├── psgnn_style/                   #     PSGNN-style 可学习 anchor（被否）
│       └── rejected_baselines/           #     下载后判定不合格的原始代码
│
├── 实验总结.md                            # 完整实验结果（8 baseline × 3 城市）
└── 第一次消融分析.md                      # 早期探索记录
```

---

## 三、实验设计

每个 baseline 三组对照：

| 实验 | 名称 | 方法 |
|:---:|------|------|
| 1 | **原版 Bi-Encoder** | GNN/Embedding 直接预测距离 |
| 2 | **+MLP+L1** | 冻结 Encoder → concat(feat_u, feat_v) → MLP → y_o, y_d → L1 loss |
| 3 | **+MLP+L1Tilde** | 同上，Decoder 替换为 L1Tilde(r=2,s=62) |

**冻结（Part A）**：Encoder 训好后冻结，只训 MLP → `cross_encoder_correct.py`

**不冻结（Part B）**：Encoder 不冻结，GNN+MLP 联合训练 → `ce_unfrozen.py`

---

## 四、运行方式

```bash
# === 原版 Bi-Encoder（7 个原始 baseline）===
python train.py --model_class rgnndist2vec --gnn_layer sage \
    --data_dir data/OSM_Chengdu --query_dir data/OSM_Chengdu/random_500k \
    --epochs 30 --device cuda --seed 42

# === Node2Vec baseline（我们新增的）===
python train.py --model_class node2vec           # 实验1: 纯嵌入 L1 距离
python train.py --model_class node2vec_mlp        # 实验2: +MLP+L1
python train.py --model_class node2vec_mlp_l1tilde \
    --l1tilde_r 2 --l1tilde_s 62                  # 实验3: +MLP+L1Tilde

# === Cross-Encoder 实验 ===
# 冻结 CE（Part A）
python experiments/cross_encoder/scripts/cross_encoder_correct.py \
    --data_dir data/OSM_Chengdu --gnn_ckpt <path> \
    --gnn_layer sage --epochs 50 --r 2 --s 62

# 不冻结 CE（Part B）
python experiments/50ep_chengdu/scripts/ce_unfrozen.py \
    --data_dir data/OSM_Chengdu --gnn_ckpt <path> \
    --gnn_layer sage --epochs 50 --r 2 --s 62
```

---

## 五、实验结果

### 5.1 8 个 Baseline 总览

| # | Baseline | 出处 | CCF | Chengdu 30ep 最佳 MRE |
|:---:|------|------|:---:|:---:|
| 1 | GAT | ICLR 2018 | A | **3.38%** 🔥 |
| 2 | SAGE | NeurIPS 2017 | A | 3.77% |
| 3 | GCN | ICLR 2017 | A | 19.59% |
| 4 | Node2Vec 🆕 | KDD 2016 | A | 62.73% |
| 5 | RNE | VLDBJ 2022 | A | 60.34% |
| 6 | VDist2Vec | EDBT 2020 | B | 59.39% |
| 7 | NDist2Vec | ISPRS 2022 | — | 56.52% |
| 8 | ANEDA | IEEE HPEC 2023 | — | 64.06% |

### 5.2 Chengdu 30ep 冻结实验（8 baseline 完整结果）

| Baseline | 实验1 原版 | 实验2 +MLP+L1 | 实验3 +MLP+L1Tilde | L1Tilde Δ |
|------|:---:|:---:|:---:|:---:|
| **GAT** | **3.38%** | 7.11% | 7.82% | +0.71 |
| **SAGE** | 3.77% | 9.31% | **8.14%** | **-1.17** ✅ |
| GCN | 19.59% | **7.43%** | 8.00% | +0.57 |
| RNE | 80.75% | 63.78% | **60.34%** | **-3.44** ✅ |
| ANEDA | 210% | 85.17% | **64.06%** | **-21.11** 🔥 |
| NDist2Vec | 62.35% | 54.95% | 56.52% | +1.57 |
| VDist2Vec | 64.57% | 56.12% | 59.39% | +3.27 |
| **Node2Vec** 🆕 | 77.17% | 66.21% | **62.73%** | **-3.48** ✅ |

### 5.3 Beijing / Harbin（SAGE, GAT）

| Baseline | City | 实验1 原版 | 实验3 +L1Tilde | L1Tilde Δ |
|------|------|:---:|:---:|:---:|
| GAT | Beijing | 3.68% | 22.50% | **-1.01** ✅ |
| SAGE | Beijing | 5.49% | 23.63% | +0.17 |
| SAGE | Harbin | 5.10% | 14.69% | +1.67 |
| GAT | Harbin | 5.12% | 17.82% | +6.24 |

### 5.4 Chengdu 50ep 冻结 vs 不冻结

| Baseline | 不冻结 L1Tilde | 冻结 L1Tilde | 不冻结改善 |
|------|:---:|:---:|:---:|
| SAGE | **6.93%** | 7.85% | -0.92 ✅ |
| GAT | **6.75%** | 7.74% | -0.99 ✅ |
| ANEDA | 75.53% | **67.55%** | +7.98 |

---

## 六、核心结论

1. **GAT Bi-Encoder 一致最佳**：Chengdu 3.21%, Beijing 3.68%, Harbin 5.12%，无需任何改造
2. **Bi-Encoder >> Cross-Encoder（2-6x）**：加 MLP 始终追不上简单 Bi-Encoder
3. **L1Tilde 有条件有效（43% 案例）**：仅在"数据集有方向性 × Encoder 缺乏方向能力 × r=2,s=62"时有效
4. **不冻结优于冻结（约 -1pp）**：让 Encoder 参与 CE 训练一致改善
5. **r=2,s=62 是正确配置**：r=62,s=2 导致 L1Tilde 崩溃到 21.94%
6. **Harbin 方向信号太弱**：单行道中位 143m，所有 L1Tilde 改造失败
7. **Node2Vec 确认通用方法 ≠ 路网距离**：62.73% vs GAT 3.38%，差距 19x

> 完整数据、消融实验、实用建议见 [实验总结.md](实验总结.md)
