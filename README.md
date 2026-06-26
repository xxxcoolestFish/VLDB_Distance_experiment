# Cross-Encoder + L1Tilde 实验

在原有 Bi-Encoder 基础上，给每个 baseline 增加 MLP Decoder，对比 L1 与 L1Tilde(r=2,s=62) 的效果。分**冻结**和**不冻结**两种训练模式。

---

## 一、代码位置

```
experiments/
├── cross_encoder/                       # 冻结实验
│   ├── models/                          #   CE 模型
│   │   ├── rgnndist2vec_mlp.py          #     GCN/GAT/SAGE + MLP + L1/L1Tilde
│   │   ├── rne_mlp.py                   #     RNE + MLP + L1/L1Tilde
│   │   ├── aneda_mlp.py                 #     ANEDA + MLP + L1/L1Tilde
│   │   ├── ndist2vec_l1tilde.py         #     NDist2Vec + L1Tilde
│   │   └── vdist2vec_l1tilde.py         #     VDist2Vec + L1Tilde
│   └── scripts/
│       └── cross_encoder_correct.py     #   冻结 CE 主脚本
│
└── 50ep_chengdu/                        # 不冻结实验
    └── scripts/
        └── ce_unfrozen.py               #   不冻结 CE 主脚本（GNN+MLP 联合训练）
```

在 `train.py` 中注册的 CE 模型（`### NEW BASELINES` 注释块）：
- `*_mlp` / `*_mlp_l1tilde` 系列 — 冻结 CE
- `node2vec` / `node2vec_mlp` / `node2vec_mlp_l1tilde` — 新增 baseline

---

## 二、实验设计

每个 baseline 三组实验：

| 实验 | 名称 | 方法 |
|:---:|------|------|
| 1 | 原版 | Bi-Encoder 直接预测距离 |
| 2 | +MLP+L1 | concat(emb_u, emb_v) → MLP → y_o, y_d → L1 loss |
| 3 | +MLP+L1Tilde | 同上，Decoder 替换为 L1Tilde(r=2,s=62) |

### 冻结（Part A）

```
阶段1: 训练 Bi-Encoder → 保存 checkpoint
阶段2: 冻结 Encoder → 提取全部节点嵌入 → 训 MLP
       只更新 MLP 参数，Encoder 不动
```

脚本：`experiments/cross_encoder/scripts/cross_encoder_correct.py`

### 不冻结（Part B）

```
阶段1: 训练 Bi-Encoder → 加载 checkpoint
阶段2: 不冻结 Encoder → GNN + MLP 联合训练
       所有参数参与反向传播
```

脚本：`experiments/50ep_chengdu/scripts/ce_unfrozen.py`

---

## 三、运行方式

```bash
# === 冻结 CE ===
python experiments/cross_encoder/scripts/cross_encoder_correct.py \
    --data_dir data/OSM_Chengdu --gnn_ckpt <ckpt_path> \
    --gnn_layer sage --epochs 50 --r 2 --s 62

# === 不冻结 CE ===
python experiments/50ep_chengdu/scripts/ce_unfrozen.py \
    --data_dir data/OSM_Chengdu --gnn_ckpt <ckpt_path> \
    --gnn_layer sage --epochs 50 --r 2 --s 62
```

---

## 四、实验结果

### Chengdu 30ep 冻结（8 个 baseline）

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

### Chengdu 50ep 不冻结（3 个 baseline）

| Baseline | +MLP+L1 | +MLP+L1Tilde | L1Tilde Δ |
|------|:---:|:---:|:---:|
| GAT | 8.54% | **6.75%** | **-1.79** ✅ |
| SAGE | 9.33% | **6.93%** | **-2.40** ✅ |
| ANEDA | 83.82% | **75.53%** | **-8.29** ✅ |

### 冻结 vs 不冻结 L1Tilde 对比

| Baseline | 冻结 | 不冻结 | 改善 |
|------|:---:|:---:|:---:|
| SAGE | 7.85% | **6.93%** | -0.92 ✅ |
| GAT | 7.74% | **6.75%** | -0.99 ✅ |
| ANEDA | **67.55%** | 75.53% | +7.98 |

### Beijing / Harbin 冻结

| Baseline | City | 原版 | +L1Tilde | Δ |
|------|------|:---:|:---:|:---:|
| GAT | Beijing | 3.68% | 22.50% | **-1.01** ✅ |
| SAGE | Beijing | 5.49% | 23.63% | +0.17 |
| SAGE | Harbin | 5.10% | 14.69% | +1.67 |
| GAT | Harbin | 5.12% | 17.82% | +6.24 |

---

## 五、核心结论

1. **GAT Bi-Encoder 一统最佳**（3.21-5.12%），CE 改造无法超越原版
2. **不冻结优于冻结（约 -1pp）**：让 Encoder 参与训练一致改善
3. **L1Tilde 有条件有效（43% 案例）**：需数据集有方向性 + Encoder 弱 + r=2,s=62
4. **r=2,s=62 是正确配置**，r=62,s=2 导致崩溃
5. **Harbin 方向信号太弱**，所有改造无效

> 完整数据见 [实验总结.md](实验总结.md)，目录说明见 [experiments/README.md](experiments/README.md)
