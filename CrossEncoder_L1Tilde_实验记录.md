# Cross-Encoder + L1Tilde 实验记录

**实验目标**: 验证"MLP concat 创造不对称 → L1Tilde 放大"在不同 baseline 上的效果

**实验协议**: 每个 baseline 2-3 组实验：
1. 原版（不加 MLP，原版 Decoder）
2. +MLP + L1（concat→MLP→y_o,y_d→L1，如已有 MLP 则跳过）
3. +MLP + L1Tilde（concat→MLP→y_o,y_d→L1Tilde）

**服务器**: 175.155.64.171

---

## 最终结果 — Chengdu (111K, 30ep, improved MLP)

| 模型 | 实验1 原版 | 实验2 +MLP+L1 | 实验3 +MLP+L1Tilde | 最佳 |
|------|:---:|:---:|:---:|:---:|
| **GAT** | **3.38%** | 14.47% | 8.76% | 原版 |
| **SAGE** | **3.77%** | 12.65% | 15.76% | 原版 |
| GCN | **19.59%** | 23.97% | 22.45% | 原版 |
| **RNE** | 80.75% | **72.42%** | 74.27% | MLP+L1 |
| **ANEDA** | 210.10% 💥 | 84.67% | **75.24%** | MLP+L1Tilde |
| NDist2Vec | 62.35% | — | 62.53% | 无差异 |
| VDist2Vec | 64.57% | — | 62.68% | 无差异 |

## 先前结果 — Harbin (43K, 50ep, simple MLP)

| 模型 | 实验1 | 实验2 | 实验3 | 最佳 |
|------|:---:|:---:|:---:|:---:|
| SAGE | 4.08% | 10.87% | 9.82% | 原版 |
| VDist2Vec | 18.78% | — | 23.37% | 原版 |

## 先前结果 — Harbin_Small (3.7K, 30ep, improved MLP)

| 模型 | 实验1 | 实验2 | 实验3 | 最佳 |
|------|:---:|:---:|:---:|:---:|
| SAGE | 9.97% | 32.55% | 19.84% | 原版 |
| GAT | 8.65% | 20.49% | 53.62% | 原版 |

---

## 核心结论

### 1. MLP concat 对 GNN Encoder 有害

SAGE/GAT/GCN 在所有数据集(Harbin/HS/Chengdu)上，MLP版本均远差于原版。
MLP 的额外参数导致过拟合，训练振荡。

### 2. MLP concat 对纯 Embedding Encoder 有帮助

RNE 和 ANEDA 在 Chengdu 上通过 MLP 显著改善。
弱 Encoder 缺乏结构先验，MLP concat 补上了这部分。

### 3. L1Tilde 效果取决于 Encoder 类型

- GAT (强 Encoder): L1Tilde > L1 within MLP (8.76% vs 14.47%)
- SAGE (中等 Encoder): L1Tilde < L1 within MLP (15.76% vs 12.65%)
- RNE (弱 Encoder): L1Tilde ≈ L1 within MLP (74.27% vs 72.42%)
- ANEDA (极弱 Encoder): L1Tilde > L1 within MLP (75.24% vs 84.67%)

### 4. 已有 Cross-Encoder 改 L1Tilde 无效

NDist2Vec, VDist2Vec 的 L1Tilde 版本与原版无差异(~62%)。
这些 baseline 本身在 Chengdu 上表现就很差。

### 5. 统一规律

```
MLP concat 补弱 Encoder 的结构缺失 → 有效
MLP concat 叠加强 Encoder → 过拟合/冗余
L1Tilde 在 MLP 内部的效果 → 取决于 Encoder 能否为不对称维度提供有效信号
```

### 6. 最佳模型

GAT 原版 Bi-Encoder: Chengdu 3.38%
