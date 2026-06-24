#!/bin/bash
# ============================================================================
# Chengdu 50-epoch 实验: SAGE, GAT, ANEDA × 两种训练模式
#  Part A: 冻结 Encoder (正确方法)
#  Part B: 不冻结 Encoder (端到端)
# ============================================================================
set +e
DATA="data/OSM_Chengdu"; QUERY="data/OSM_Chengdu/random_500k"
EP=50; BS=1024; EMB=64; LR=0.001; LOSS=smoothl1; SEED=42; R=2; S=62
PY="/root/miniconda3/bin/python -u train.py"
CE_CORRECT="experiments/cross_encoder/scripts/cross_encoder_correct.py"
CE_UNFROZEN="experiments/50ep_chengdu/scripts/ce_unfrozen.py"
CE_EMB="experiments/cross_encoder/scripts/cross_encoder_emb.py"
RESULTS="results/50ep_chengdu"

# ============================================================================
# Part A: 冻结 Encoder — 先训 Bi-Encoder 50ep, 再冻住跑 CE
# ============================================================================
for model in sage gat aneda; do
    echo ""
    echo "============================================================"
    echo "  Part A: $model — Train Bi-Encoder 50ep (frozen)"
    echo "============================================================"

    case $model in
        sage|gat)
            CKPT="$RESULTS/${model}_bie/saved_models/rgnndist2vec_OSM_Chengdu_random_500k.pt"
            if [ ! -f "$CKPT" ]; then
                $PY --model_class rgnndist2vec --gnn_layer $model \
                    --data_dir $DATA --query_dir $QUERY --epochs $EP --device cuda \
                    --batch_size_train $BS --embedding_dim $EMB --loss $LOSS \
                    --learning_rate $LR --seed $SEED --validate --force_shift 0 \
                    --log_dir "$RESULTS/${model}_bie"
            else
                echo "[SKIP] $model Bi-Encoder already trained"
            fi

            echo "--- $model CE (Frozen) ---"
            python $CE_CORRECT --data_dir $DATA --gnn_ckpt "$CKPT" \
                --gnn_layer $model --epochs $EP --r $R --s $S
            ;;
        aneda)
            CKPT="$RESULTS/${model}_bie/saved_models/${model}_OSM_Chengdu_random_500k.pt"
            if [ ! -f "$CKPT" ]; then
                $PY --model_class aneda --data_dir $DATA --query_dir $QUERY \
                    --epochs $EP --device cuda --batch_size_train $BS \
                    --embedding_dim $EMB --loss $LOSS --learning_rate $LR \
                    --seed $SEED --validate --force_shift 0 \
                    --log_dir "$RESULTS/${model}_bie"
            else
                echo "[SKIP] $model Bi-Encoder already trained"
            fi

            echo "--- $model CE (Frozen) ---"
            python $CE_EMB --data_dir $DATA --model_class aneda --ckpt "$CKPT" \
                --epochs $EP --r $R --s $S
            ;;
    esac
done

# ============================================================================
# Part B: 不冻结 Encoder — 从 Bi-Encoder 初始化, 端到端训练 CE
# ============================================================================
for model in sage gat aneda; do
    echo ""
    echo "============================================================"
    echo "  Part B: $model — Unfrozen CE (end-to-end, 50ep)"
    echo "============================================================"

    case $model in
        sage|gat)
            CKPT="$RESULTS/${model}_bie/saved_models/rgnndist2vec_OSM_Chengdu_random_500k.pt"
            python $CE_UNFROZEN --data_dir $DATA --model_class $model \
                --gnn_ckpt "$CKPT" --epochs $EP --r $R --s $S
            ;;
        aneda)
            CKPT="$RESULTS/${model}_bie/saved_models/${model}_OSM_Chengdu_random_500k.pt"
            python $CE_UNFROZEN --data_dir $DATA --model_class aneda \
                --gnn_ckpt "$CKPT" --epochs $EP --r $R --s $S
            ;;
    esac
done

echo ""
echo "===== ALL DONE ====="
