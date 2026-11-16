#  Response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts

This repository contains code for analyzing single-cell perturbation prediction as a response component prediction problem.

The analyses decompose perturbation responses into shared effects, perturbation-specific residuals, conserved perturbation components, and cell-line-by-perturbation interactions. The repository also includes code for DepMap-based prediction, cross-cell-line transfer, combinatorial perturbation prediction, chemical perturbation transfer in Tahoe-100M, pathway-level evaluation, and prediction component attribution.

This repository is under active development for manuscript reproducibility.


> **Check out the preprint**
>
> This repository contains the reproducibility code. For the biological motivation, full results and interpretation, you can read the paper here:  
> https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1

## Main analyses

1. Construct pseudobulk perturbation responses.
2. Compute perturbed-reference metrics.
3. Decompose responses into template, residual, conserved perturbation, and interaction components.
4. Evaluate DepMap response-aligned Ridge and MLP models.
5. Evaluate cross-cell-line transfer and target calibration.
6. Analyze combinatorial perturbation prediction on Norman et al.
7. Analyze chemical perturbation transfer in Tahoe-100M.
8. Attribute model predictions to response components.

## Quick start

```bash
# 1. Install
git clone https://github.com/xinyizhanglab/perturbation-decomposition
cd perturbation-decomposition
conda env create -f environment.yml
conda activate pertdecomp

# 2. Data is included (processed pseudobulk + DepMap embeddings, ~90 MB)
# Pre-computed predictions for all models are in results/ (~150 MB)

# 3. Run baselines from scratch (optional, ~10 min per CL on CPU)
python3 -m benchmarks.within_cl --model ridge --cell_line all \
  --data_dir data/processed/genetic \
  --depmap data/processed/genetic/depmap_embeddings.pkl

# 4. Evaluate any prediction file
python3 -c "
from evaluation.metrics import evaluate_predictions
print(evaluate_predictions('results/within_cl/ridge/k562/fold0/predictions.npz'))
"
```

## Pre-computed results

All predictions are included in `results/`. No retraining needed to reproduce figures or tables. These can be run in any desktop computer.

| Directory | Models | Seeds/Folds |
|-----------|--------|-------------|
| `results/within_cl/ridge/` | Ridge 5-fold CV | 4 CL x 5 folds |
| `results/within_cl/mlp/` | MLP 5-fold CV | 4 CL x 5 folds |
| `results/within_cl/morph/` | MORPH 5-fold CV | 4 CL x 5 folds |
| `results/within_cl/scgpt/` | scGPT | 4 CL x 5 seeds |
| `results/cross_cl/ridge/` | Ridge 3-to-1 transfer | 4 targets |
| `results/cross_cl/mlp/` | MLP 3-to-1 transfer | 4 targets |
| `results/cross_cl/morph/` | MORPH 1-to-1 zeroshot + finetuned + 30% | 4 pairs x 5 seeds x 3 settings |
| `results/cross_cl/state_zeroshot/` | STATE zeroshot (target excluded) | 4 CL |
| `results/cross_cl/state_crosscl/` | STATE +30% target | 4 CL x 5 seeds |
| `results/norman/` | Norman double-gene benchmark | Ridge, MLP, MORPH |
| `results/tahoe/` | Tahoe drug benchmark | Morgan, CC, source transfer |
| `results/decomposition/` | ANOVA + template/residual attribution | |
| `results/calibration/` | STATE/MORPH calibration analysis | |
| `results/pathway/` | Hallmark pathway prediction | |

## Benchmarks

### 1. Within-cell-line prediction (5-fold CV)

Hold out 20% of perturbations, predict from DepMap dependency profiles.

```bash
python3 -m benchmarks.within_cl --model ridge --cell_line k562 \
  --data_dir data/processed/genetic \
  --depmap data/processed/genetic/depmap_embeddings.pkl
```

### 2. Cross-cell-line transfer (zero-shot)

Train on source CL(s), predict target CL without any target training data.

```bash
python3 -m benchmarks.cross_cl --model ridge --target k562 --n_sources 3 \
  --data_dir data/processed/genetic \
  --depmap data/processed/genetic/depmap_embeddings.pkl
```

### 3. Norman combinatorial perturbations

Predict double-gene responses from DepMap pair features. Stratified by unseen gene count (0/2, 1/2, 2/2).

```bash
python3 -m benchmarks.norman_doubles --data_dir data/processed/norman
```

### 4. Tahoe drug perturbations

Compare compound priors (Morgan fingerprints, Chemical Checker) against source-response transfer. Includes MoA-held-out evaluation and full testing battery.

```bash
python3 -m benchmarks.tahoe_drugs --data_dir data/processed/tahoe --features all
python3 -m benchmarks.tahoe_drugs --data_dir data/processed/tahoe --features source --rank_sweep
python3 -m benchmarks.tahoe_drugs --data_dir data/processed/tahoe --moa_holdout
```

### 5. ANOVA decomposition

Four-component decomposition of perturbation responses.

```
delta = mu + alpha_c + beta_p + gamma_{c,p}
```

where mu is the global mean, alpha is the cell-line effect, beta is the conserved perturbation effect, and gamma is the cell-line by perturbation interaction.

```bash
python3 -m decomposition.anova --data_dir data/processed/genetic
```

### 6. Pathway-level prediction

Per-pathway prediction accuracy using MSigDB Hallmark gene sets.

```bash
python3 -m benchmarks.pathway_prediction \
  --data_dir data/processed/genetic \
  --depmap data/processed/genetic/depmap_embeddings.pkl
```

## Models

| Model | Type | Features | Within-CL | Cross-CL | Code |
|-------|------|----------|-----------|----------|------|
| Ridge | Linear | DepMap PCA | Yes | Yes | `models/ridge.py` |
| MLP | Neural (2 hidden) | DepMap PCA | Yes | Yes | `models/ridge.py` |
| MORPH | Conditional VAE | DepMap (encoder) | Yes | Yes | `models/model_scripts/` |
| STATE | Set-attention | One-hot | No | Yes | `models/model_scripts/` |
| scGPT | Transformer | Pretrained embeddings | Yes | No | `models/model_scripts/` |


### External model setup

MORPH, STATE, and scGPT require their original codebases. Training and evaluation code is provided. Retraining these models can take up between 2 to 48 hours in a single Nvidia Blackwell GPU.


## Metrics

| Metric | Description | Primary |
|--------|-------------|---------|
| pref | Template-removed Pearson: corr(delta - T_train, delta_hat - T_train) | Yes |
| std_r | Standard Pearson: corr(delta, delta_hat) | No |



## Data

Processed data is included in `data/processed/` (~90 MB total). Raw data can be regenerated with `data/prepare_release.py`.

| Dataset | Cell lines | Perturbations | Type | File |
|---------|-----------|---------------|------|------|
| Replogle 2022 | K562, RPE1 | ~1300 each | CRISPRi | `genetic/{k562,rpe1}.pkl` |
| Nadig 2024 | HepG2, Jurkat | ~1300 each | CRISPRi | `genetic/{hepg2,jurkat}.pkl` |
| Norman 2019 | K562 | 105 singles + 131 doubles | CRISPRa | `norman/norman.pkl` |
| Tahoe-100M | 5 CLs | 268 drugs x 3 doses | Small molecule | `tahoe/pseudobulk_deltas.pkl` |
| DepMap | 1208 CLs | 18531 genes | Dependency | `genetic/depmap_embeddings.pkl` |

## Repository structure

```
perturbation-decomposition/
├── configs/datasets.yaml          # all hyperparameters
├── data/
│   ├── process.py                 # raw h5ad -> pseudobulk
│   ├── prepare_release.py         # cluster -> release files
│   └── processed/                 # included: pseudobulk + embeddings
├── decomposition/
│   └── anova.py                   # 4-component ANOVA
├── models/
│   ├── ridge.py                   # Ridge + MLP baselines
│   ├── morph/                     # MORPH wrappers
│   ├── state/                     # STATE wrappers
│   └── scgpt/                     # scGPT evaluation
├── benchmarks/
│   ├── within_cl.py               # 5-fold CV
│   ├── cross_cl.py                # zero-shot transfer
│   ├── norman_doubles.py          # combinatorial benchmark
│   ├── tahoe_drugs.py             # drug benchmark + testing battery
│   └── pathway_prediction.py      # Hallmark pathway analysis
├── evaluation/
│   └── metrics.py                 # pref, std_r, save/load
├── figures/                       # all paper figure scripts
└── results/                       # pre-computed predictions + analyses
```

## Citation

## Citation

If you use this work, please cite:

Molina, Alexis, and Xinyi Zhang. “Perturbation response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts.” *bioRxiv* (2026): 2026-07.

```bibtex
@article{Molina2026Perturbation,
  author  = {Molina, Alexis and Zhang, Xinyi},
  title   = {Perturbation response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts},
  journal = {bioRxiv},
  year    = {2026},
  note    = {Preprint}
}
```

## License

MIT