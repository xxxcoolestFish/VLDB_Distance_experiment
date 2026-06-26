# 新 Baseline 实验

在原有 7 个 baseline 基础上，新增 **Node2Vec [KDD 2016, CCF-A]** 作为第 8 个 baseline。

---

## 一、新增代码位置

```
experiments/new_baselines/
└── node2vec/
    └── node2vec_baseline.py    # Node2Vec baseline（含三组实验）
```

`train.py` 中对应注册了三个 `model_class`：
- `node2vec` — 实验1：纯嵌入 L1 距离
- `node2vec_mlp` — 实验2：+MLP+L1
- `node2vec_mlp_l1tilde` — 实验3：+MLP+L1Tilde(r=2,s=62)

---

## 二、实验设计

| 实验 | model_class | 方法 |
|:---:|------|------|
| 1 | `node2vec` | Node2Vec 随机游走+skip-gram 生成嵌入 → L1 距离 = 预测距离（无训练） |
| 2 | `node2vec_mlp` | 冻结 Node2Vec 嵌入 → concat → MLP → L1 loss |
| 3 | `node2vec_mlp_l1tilde` | 同上，Decoder 替换为 L1Tilde(r=2,s=62) |

---

## 三、运行方式

```bash
# 首次运行会自动生成 Node2Vec 嵌入（data/OSM_Chengdu/node2vec_dim64.npy）
# 之后直接加载缓存

python train.py --model_class node2vec \
    --data_dir data/OSM_Chengdu --query_dir data/OSM_Chengdu/random_500k \
    --embedding_dim 64 --epochs 30 --device cuda --seed 42

python train.py --model_class node2vec_mlp \
    --data_dir data/OSM_Chengdu --query_dir data/OSM_Chengdu/random_500k \
    --embedding_dim 64 --epochs 30 --device cuda --seed 42

python train.py --model_class node2vec_mlp_l1tilde \
    --l1tilde_r 2 --l1tilde_s 62 \
    --data_dir data/OSM_Chengdu --query_dir data/OSM_Chengdu/random_500k \
    --embedding_dim 64 --epochs 30 --device cuda --seed 42
```

---

## 四、实验结果（Chengdu 30ep）

| Baseline | 出处 | 实验1 原版 | 实验2 +MLP+L1 | 实验3 +MLP+L1Tilde | L1Tilde Δ |
|------|------|:---:|:---:|:---:|:---:|
| **Node2Vec** 🆕 | KDD 2016 | 77.17% | 66.21% | **62.73%** | **-3.48** ✅ |

与其他 baseline 对比：

| # | Baseline | 出处 | CCF | 最佳 MRE |
|:---:|------|------|:---:|:---:|
| 1 | GAT | ICLR 2018 | A | **3.38%** 🔥 |
| 2 | SAGE | NeurIPS 2017 | A | 3.77% |
| 3 | GCN | ICLR 2017 | A | 7.43% |
| 4 | NDist2Vec | ISPRS 2022 | — | 56.52% |
| 5 | VDist2Vec | EDBT 2020 | B | 59.39% |
| 6 | RNE | VLDBJ 2022 | A | 60.34% |
| 7 | **Node2Vec** 🆕 | KDD 2016 | A | 62.73% |
| 8 | ANEDA | IEEE HPEC 2023 | — | 64.06% |

---

## 五、如何添加新 Baseline

1. 在 `experiments/new_baselines/<name>/` 下写模型代码
2. 在 `train.py` 中添加 `elif model_class == 'xxx':` 分支
3. 保持 `from experiments.new_baselines.xxx import ...` 路径，不修改 `models/`

> 完整 Cross-Encoder 实验（冻结/不冻结/L1Tilde）见 `experiments/README.md` 和 `实验总结.md`
