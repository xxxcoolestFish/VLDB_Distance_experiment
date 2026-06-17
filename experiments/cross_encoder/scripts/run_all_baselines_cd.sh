#!/bin/bash
# ============================================================================
# 全 Baseline Cross-Encoder + L1Tilde 实验 — Chengdu (111K nodes)
# 每个baseline三组实验: 1=原版, 2=+MLP+L1, 3=+MLP+L1Tilde
# ============================================================================
DATA="data/OSM_Chengdu"
QUERY="data/OSM_Chengdu/random_500k"
EP=30; BS=1024; EMB=64; LR=0.001; LOSS=smoothl1; SEED=42
R=62; S=2
PY="/root/miniconda3/bin/python -u train.py"
COMMON="--data_dir $DATA --query_dir $QUERY --epochs $EP --device cuda --force_shift 0 --batch_size_train $BS --embedding_dim $EMB --loss $LOSS --learning_rate $LR --seed $SEED --validate"

run_exp() {
    local name=$1 model=$2 extra=$3
    local log="results/cd_${name}"
    if [ -f "$log/experiment_results.json" ]; then
        echo "[SKIP] $name — already done"
    else
        echo "===== $name ====="
        $PY $model --log_dir $log $COMMON $extra 2>&1 | grep "Mean Relative Error\|Global MRE" | tail -2
    fi
}

# ============ GNN-based ============
for gnn in sage gat gcn; do
    run_exp "${gnn}_exp1"      "--model_class rgnndist2vec --gnn_layer $gnn" ""
    run_exp "${gnn}_exp2"      "--model_class rgnndist2vec_mlp --gnn_layer $gnn" ""
    run_exp "${gnn}_exp3"      "--model_class rgnndist2vec_mlp_l1tilde --gnn_layer $gnn --l1tilde_r $R --l1tilde_s $S" ""
done

# ============ RNE ============
run_exp "rne_exp1"             "--model_class rne"             "--loss mse --learning_rate 0.003"
run_exp "rne_exp2"             "--model_class rne_mlp"         "--loss mse --learning_rate 0.003"
run_exp "rne_exp3"             "--model_class rne_mlp_l1tilde --l1tilde_r $R --l1tilde_s $S --loss mse --learning_rate 0.003"

# ============ ANEDA ============
run_exp "aneda_exp1"           "--model_class aneda"           ""
run_exp "aneda_exp2"           "--model_class aneda_mlp"       ""
run_exp "aneda_exp3"           "--model_class aneda_mlp_l1tilde --l1tilde_r $R --l1tilde_s $S"

# ============ NDist2Vec (exp2 skipped — already has MLP) ============
run_exp "ndist2vec_exp1"       "--model_class ndist2vec"       ""
run_exp "ndist2vec_exp3"       "--model_class ndist2vec_l1tilde --l1tilde_r $R --l1tilde_s $S"

# ============ VDist2Vec (exp2 skipped — already has MLP) ============
run_exp "vdist2vec_exp1"       "--model_class vdist2vec"       ""
run_exp "vdist2vec_exp3"       "--model_class vdist2vec_l1tilde --l1tilde_r $R --l1tilde_s $S"

echo "===== ALL DONE ====="
