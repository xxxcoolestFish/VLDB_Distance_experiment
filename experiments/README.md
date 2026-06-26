# 实验代码目录

本文档说明 `experiments/` 目录结构，区分**我们的新实验**与**原始项目代码**。

---

## 目录结构

```
experiments/
├── README.md                          # 本文件
│
├── cross_encoder/                     # 🟢 Cross-Encoder 实验（核心工作）
│   ├── models/                        # CE 模型变体：冻结 encoder + MLP → L1/L1Tilde
│   │   ├── rgnndist2vec_mlp.py       #   GCN/GAT/SAGE + MLP + L1/L1Tilde
│   │   ├── rne_mlp.py                #   RNE + MLP + L1/L1Tilde
│   │   ├── aneda_mlp.py              #   ANEDA + MLP + L1/L1Tilde
│   │   ├── ndist2vec_l1tilde.py      #   NDist2Vec + L1Tilde
│   │   └── vdist2vec_l1tilde.py      #   VDist2Vec + L1Tilde
│   └── scripts/                       # CE 训练脚本
│       ├── cross_encoder_correct.py   #   冻结 CE 主脚本（Part A）
│       └── cross_encoder_emb.py       #   嵌入预计算工具
│
├── 50ep_chengdu/                      # 🟢 50 epoch 实验（Chengdu 冻结 vs 不冻结）
│   ├── scripts/
│   │   ├── ce_unfrozen.py            #   不冻结 CE 脚本（Part B：GNN+MLP 联合训练）
│   │   └── run_all.sh                #   一键运行
│   └── 实验结果.md                    #   50ep 实验结果记录
│
├── new_baselines/                     # 🟢 我们新增的 baseline
│   └── node2vec/
│       └── node2vec_baseline.py       #   Node2Vec [KDD 2016] — 含 pure / +MLP / +L1Tilde 三模式
│
├── archived/                          # 🗂️ 已废弃的实验（仅供参考）
│   ├── dlg_style/                     #   DLGNN-style anchor 增强（被否：原始任务非距离估计）
│   ├── psgnn_style/                   #   PSGNN-style 可学习 anchor（被否：同上）
│   └── rejected_baselines/            #   下载后判定不合格的 baseline 原始代码
│       ├── DLGNN/                     #   链接预测，非距离估计
│       ├── PSGNN/                     #   节点分类，非距离估计
│       ├── GRED/                      #   图回归，非距离估计
│       └── tgt/                       #   分子图距离，非路网
│
└── __init__.py
```

---

## 与原始代码的关系

| 目录 | 说明 |
|------|------|
| `models/` （根目录） | **原始项目代码**，未修改 |
| `utils/` （根目录） | **原始工具函数**，未修改 |
| `original/` （根目录） | 原始代码的备份 |
| `train.py` （根目录） | 原始训练入口，**我们添加了新模型注册**（在 `### NEW BASELINES` 注释块中标记） |
| `experiments/` | **我们的全部新代码**，独立于原始代码 |

**原则：所有新代码都在 `experiments/` 下，不覆写原始 `models/`。**

---

## 实验协议

每个 baseline 跑三组实验：

| 实验 | 名称 | 方法 |
|:---:|------|------|
| 1 | 原版 | Bi-Encoder 直接预测距离 |
| 2 | +MLP+L1 | 冻结 Encoder → concat → MLP → L1 loss |
| 3 | +MLP+L1Tilde | 同上，L1 替换为 L1Tilde(r=2,s=62) |

**冻结（Part A）**：Encoder 训好后冻结，只训 MLP。参见 `cross_encoder/scripts/cross_encoder_correct.py`

**不冻结（Part B）**：Encoder 不冻结，GNN+MLP 联合训练。参见 `50ep_chengdu/scripts/ce_unfrozen.py`

---

## 运行方式

```bash
# Node2Vec baseline（三组实验）
python train.py --model_class node2vec          # 实验1: pure
python train.py --model_class node2vec_mlp       # 实验2: +MLP+L1
python train.py --model_class node2vec_mlp_l1tilde --l1tilde_r 2 --l1tilde_s 62  # 实验3

# 原始 baseline + CE（冻结）
python cross_encoder/scripts/cross_encoder_correct.py --model_class sage --use_l1tilde --l1tilde_r 2 --l1tilde_s 62

# 不冻结 CE
python 50ep_chengdu/scripts/ce_unfrozen.py --model_class sage --l1tilde_r 2 --l1tilde_s 62
```

---

## 8 个 Baseline 总览

| # | Baseline | 出处 | CCF | 类型 |
|:---:|------|------|:---:|------|
| 1 | GAT | ICLR 2018 | A | 通用 GNN |
| 2 | SAGE | NeurIPS 2017 | A | 通用 GNN |
| 3 | GCN | ICLR 2017 | A | 通用 GNN |
| 4 | Node2Vec 🆕 | KDD 2016 | A | 通用图嵌入 |
| 5 | VDist2Vec | EDBT 2020 | B | 路网距离专用 |
| 6 | NDist2Vec | ISPRS 2022 | — | 路网距离专用 |
| 7 | RNE | VLDBJ 2022 | A | 路网距离专用 |
| 8 | ANEDA | IEEE HPEC 2023 | — | 路网距离专用 |
