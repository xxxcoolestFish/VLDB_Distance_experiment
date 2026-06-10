# VLDB Distance Experiment — Project Guide

## Project Overview

L̃₁ asymmetric metric ablation study for directed road network shortest-path distance estimation.
Experiments compare L1 vs L̃₁ across 5 baselines (RGAT, RSAGE, RGCN, RNE, LpNorm).

Key finding: L̃₁ helps RSAGE on large directed graphs (-1.4% to -1.9% MRE), but GAT's attention
already captures directionality — the metric's benefit depends on architecture.

## Remote Server

```
Host:     175.155.64.171
Port:     24186
User:     root
SSH:      sshpass -p '<password>' ssh -p 24186 root@175.155.64.171
Project:  /root/mornai-tmp/VLDB_Distance_experiment
Data:     /root/mornai-tmp/VLDB_Distance/shortest-distance-survey/data -> symlinked as data/
```

**Environment** (conda base): Python 3.10.12, PyTorch 2.5.1+cu124, CUDA 12.4, RTX 3090 (24 GB)

**Run commands on server with**: `sshpass -p '<password>' ssh -p 24186 root@175.155.64.171 '<command>'`

**Python on server**: `~/miniconda3/bin/python`

## Workflow Rules (CRITICAL)

1. **Code changes happen locally first** — edit files in this local workspace, never directly on the server
2. **Sync to server** — after local changes, upload to server via scp/rsync:
   ```bash
   # Quick sync of changed files:
   tar czf /tmp/vldb_sync.tar.gz <changed-files> && \
   sshpass -p '<password>' scp -P 24186 /tmp/vldb_sync.tar.gz root@175.155.64.171:/tmp/ && \
   sshpass -p '<password>' ssh -p 24186 root@175.155.64.171 \
     'cd ~/mornai-tmp/VLDB_Distance_experiment && tar xzf /tmp/vldb_sync.tar.gz && rm /tmp/vldb_sync.tar.gz'
   ```
3. **Run experiments on server** — GPU training only happens on the remote server

## Experiment Scope

- **ONLY use Harbin data** for testing and experiments (`OSM_Harbin`, 44K nodes, 108K edges)
- Other cities (Beijing, Chengdu, Qingdao, Harbin_Small) are NOT considered for now — fast iteration
- Run full benchmark across all 4 cities only when a method is proven on Harbin

## Quick Commands

### Test a single model on Harbin (on server):
```bash
sshpass -p '<password>' ssh -p 24186 root@175.155.64.171 \
  'cd ~/mornai-tmp/VLDB_Distance_experiment && \
   ~/miniconda3/bin/python -u train.py \
     --model_class rgnndist2vec --gnn_layer gat \
     --data_dir data/OSM_Harbin --query_dir data/OSM_Harbin/proportional \
     --epochs 20 --device cuda --force_shift 0 \
     --log_dir results/test_harbin_rgat'
```

### Data available on server (all cities have: .nodes, .edges, .queries, .parts, landmark_dim61.embeddings, dist2vec_dim62.npy):
```
data/OSM_Harbin/          (117M)
data/OSM_Harbin_Small/    (21M, reserved)
data/OSM_Chengdu/         (280M, skip for now)
data/OSM_Qingdao/         (303M, skip for now)
data/OSM_Beijing/         (412M, skip for now)
```

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main entry point, supports all 20+ models via `--model_class` |
| `models/basemodel.py` | BaseModel class with `fit()` / `evaluate()` |
| `utils/asymmetric_metrics.py` | L̃₁ / L̃∞ metric implementations |
| `utils/data_utils.py` | Graph I/O, landmark selection, `ensure_*` auto-generation |
| `utils/torch_utils.py` | Dataset classes, optimizer, device detection |
| `scripts/prepare_data.py` | Download + convert OSM data |
| `scripts/run_full_benchmark.sh` | Full benchmark: 16 baselines × 5 cities |
| `scripts/run_l1tilde_ablation.sh` | Ablation: 5 L1Tilde models × 5 cities |

## Supported Models (for reference)

**L1 baselines:** rgnndist2vec (--gnn_layer gat/sage/gcn), rne, lpnorm
**L̃₁ variants:** rgnndist2vec_l1tilde, rne_l1tilde, lpnorm_l1tilde
**Other baselines:** geodnn, ndist2vec, vdist2vec, distancenn, embeddingnn, aneda, path2vec, catboost, catboostnn, landmark
**Our method:** dist2gnn
