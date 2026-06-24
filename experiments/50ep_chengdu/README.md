# Chengdu 50-Epoch 实验

两个实验目标：
1. 验证 50 epoch 训练是否改善所有变体（Bi-Encoder / CE+L1 / CE+L1Tilde）
2. 验证不冻结 Encoder 的端到端训练是否优于冻结的

## Baseline: SAGE, GAT, ANEDA

## Part A: 冻结 Encoder（正确方法，50ep）

每个 baseline:
- 训练 Bi-Encoder 50epoch → 保存 checkpoint
- 冻结 Encoder → CE+L1 (50ep) / CE+L1Tilde (50ep)

## Part B: 不冻结 Encoder（端到端）

每个 baseline:
- 从 Part A 的 Bi-Encoder checkpoint 初始化
- Encoder 参数不冻结 (requires_grad=True)
- GNN+MLP 一起训练 50epoch
- 对比 CE+L1 vs CE+L1Tilde

## 运行

```bash
bash experiments/50ep_chengdu/scripts/run_all.sh
```

## 预期对比

| 实验 | Encoder | 训练方式 | 与之前(30ep)对比 |
|------|:---:|------|------|
| Part A Bi-Enc | 50ep 训练 | 单阶段 | 看 50ep 是否改善 Bi-Encoder |
| Part A CE+L1 | 50ep 冻结 | 两阶段 | 看更多 CE epochs 是否改善 |
| Part A CE+L1Tilde | 50ep 冻结 | 两阶段 | 同上 |
| Part B CE+L1 | 50ep 端到端 | 解冻 | vs Part A: 解冻是否更好 |
| Part B CE+L1Tilde | 50ep 端到端 | 解冻 | vs Part A: L1Tilde 在端到端是否有效 |
