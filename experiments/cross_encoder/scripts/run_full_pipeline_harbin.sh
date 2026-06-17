#!/bin/bash
# ============================================================================
# 完整实验 Pipeline — Harbin (r=2,s=62)
# 阶段1: 训练 Bi-Encoder (GNN) → 保存 checkpoint
# 阶段2: 冻结 GNN → Cross-Encoder L1 vs L1Tilde
# ============================================================================
DATA="data/OSM_Harbin"
QUERY="data/OSM_Harbin/random_500k"
EP=30; BS=1024; EMB=64; LR=0.001; LOSS=smoothl1; SEED=42
R=2; S=62
PY="/root/miniconda3/bin/python -u train.py"

# ============ SAGE ============
echo "===== Stage1: Train SAGE Bi-Encoder ====="
$PY --model_class rgnndist2vec --gnn_layer sage \
    --data_dir $DATA --query_dir $QUERY --epochs $EP --device cuda --force_shift 0 \
    --batch_size_train $BS --embedding_dim $EMB --loss $LOSS --learning_rate $LR --seed $SEED \
    --validate --log_dir results/hb_sage_stage1

echo "===== Stage2: SAGE Cross-Encoder L1 vs L1Tilde ====="
/root/miniconda3/bin/python -u experiments/cross_encoder/scripts/cross_encoder_correct.py \
    --data_dir $DATA --gnn_ckpt results/hb_sage_stage1/saved_models/rgnndist2vec_OSM_Harbin_random_500k.pt \
    --gnn_layer sage --epochs 50 --r $R --s $S

# ============ GAT ============
echo "===== Stage1: Train GAT Bi-Encoder ====="
$PY --model_class rgnndist2vec --gnn_layer gat \
    --data_dir $DATA --query_dir $QUERY --epochs $EP --device cuda --force_shift 0 \
    --batch_size_train $BS --embedding_dim $EMB --loss $LOSS --learning_rate $LR --seed $SEED \
    --validate --log_dir results/hb_gat_stage1

echo "===== Stage2: GAT Cross-Encoder L1 vs L1Tilde ====="
/root/miniconda3/bin/python -u experiments/cross_encoder/scripts/cross_encoder_correct.py \
    --data_dir $DATA --gnn_ckpt results/hb_gat_stage1/saved_models/rgnndist2vec_OSM_Harbin_random_500k.pt \
    --gnn_layer gat --epochs 50 --r $R --s $S

echo "===== ALL DONE ====="
