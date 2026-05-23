# VLDB Distance

Code and experiments for: *"Does L̃₁ asymmetric metric improve over L1 for directed road network distance estimation?"*

## Directory Structure

```
├── original/              # All 16 baseline + Dist2GNN source code (unchanged)
│   ├── train.py           #   Training script
│   ├── basemodel.py       #   Base model class
│   ├── geodnn.py, vdist2vec.py, ndist2vec.py, ...
│   ├── rgnndist2vec.py    #   ★ RGAT/RSAGE/RGCN — uses L1 norm
│   ├── rne.py             #   ★ RNE — uses L1-mean
│   ├── lpnorm.py          #   ★ LpNorm — uses L1 (Manhattan)
│   ├── dist2gnn_model.py  #   ★ Dist2GNN — uses L̃₁ (our method)
│   ├── asymmetric_metrics.py  # L̃₁/L̃∞ definition
│   ├── dist2vec.py, active_finetune.py
│   └── data_utils.py, torch_utils.py, plot_utils.py
│
├── modified/              # L1 → L̃₁ modified files + diffs
│   ├── rgnndist2vec_l1tilde.py   # RGAT/RSAGE/RGCN → L̃₁
│   ├── rne_l1tilde.py            # RNE → L̃₁
│   ├── lpnorm_l1tilde.py         # LpNorm → L̃₁
│   ├── train.py                  # Training script (+5 model_class branches)
│   └── *.diff                    # Exact line-by-line changes
│
└── scripts/               # Experiment orchestration
    ├── run_full_benchmark.sh     # Full: 16 baselines × 5 cities = 85 exps
    ├── run_l1tilde_ablation.sh   # Ablation: 5 L1Tilde models × 5 cities
    └── collect_results.py        # Collect & compare L1 vs L1Tilde
```

## Experiment Design

**Question**: What is the real contribution of L̃₁ over L1, controlling for model architecture?

**Method**: Take 5 baselines that use L1 distance (RGAT, RSAGE, RGCN, RNE, LpNorm), create L̃₁ variants with *identical architecture*, run the same 5-city directed road network protocol, measure ΔMRE.

**Theoretical basis**: Theorem 3-5 from the paper prove L1 cannot isometrically embed directed graphs, but L̃₁ can.

## Key Finding

| Model | Small cities | Large cities | Overall |
|-------|:---:|:---:|:---:|
| RGAT→L1Tilde | ~same | ~same | No significant gain |
| RSAGE→L1Tilde | ~same | **-1.4 to -1.9%** | Improves on large graphs |
| RGCN→L1Tilde | ~same | ~same | No change |
| RNE→L1Tilde | -1% | **+12 to +16%** | Degrades |
| LpNorm→L1Tilde | broken | broken | N/A (raw coords) |

**Conclusion**: L̃₁ helps SAGE (which lacks attention) on large directed graphs, but GAT's attention mechanism already captures directionality. The metric's benefit depends on the architecture's ability to utilize asymmetric dimensions.

## Usage

### 1. Run full benchmark (original baselines)
```bash
cd scripts/
# Edit paths in run_full_benchmark.sh first
bash run_full_benchmark.sh
```

### 2. Run L1Tilde ablation
```bash
# First: copy modified/ files into your survey src/models/
cp modified/*.py /path/to/survey/src/models/
cp modified/train.py /path/to/survey/src/

# Then run:
bash run_l1tilde_ablation.sh
```

### 3. Collect & compare results
```bash
python3 collect_results.py --base /path/to/results
```

## Modifications (per baseline, ~10 lines each)

The change is minimal and localized:

1. `__init__`: add `l1tilde_r`, `l1tilde_s` parameters
2. `forward()`: replace `torch.norm(emb1-emb2, p=1)` with:
   ```python
   r, s = self.l1tilde_r, self.l1tilde_s
   sym = torch.abs(emb2[:,:r] - emb1[:,:r]).sum(dim=1, keepdim=True)
   asym = (emb2[:,r:r+s] - emb1[:,r:r+s]).sum(dim=1, keepdim=True)
   distance = sym + asym
   ```

See `modified/*.diff` for exact line-by-line changes.
