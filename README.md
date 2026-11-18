#  Response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts

This repository contains code for analyzing single-cell perturbation prediction as a response component prediction problem.

The analyses decompose perturbation responses into shared effects, perturbation-specific residuals, conserved perturbation components, and cell-line-by-perturbation interactions. The repository also includes code for DepMap-based prediction, cross-cell-line transfer, combinatorial perturbation prediction, chemical perturbation transfer in Tahoe-100M, pathway-level evaluation, and prediction component attribution.

Includes **DepMap Ridge/MLP, MORPH, STATE, scGPT, BioLORD, and scGen**. For model-based results, use **saved predictions → inference → retraining**. Figures that only use saved data need no model execution.


> **Check out the preprint**
>
> This repository contains the reproducibility code. For the biological motivation, full results and interpretation, you can read the paper here:  
> https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1

## Quick start

```bash
uv sync --locked --python 3.10
uv run python -m data.check
```

Run from the repository root. Start with Data, then regenerate figures from
saved results below. Model workflows follow **saved predictions → inference → retraining**.
Use new output directories. On the cluster, substantial analysis, inference, and
training run inside compute allocations.

## Data

Figure inputs and processed response data are included in `reproduction_files/`.
Run from the repository root:

```bash
RESULTS_DOWNLOADS=reproduction_files
DATA_DOWNLOADS=reproduction_files
```

No separate data download is needed for the supported saved-figure commands.
Large prediction, weight, and native-training files are optional and are not
included in the source checkout. They are available in the
[`rep4rev-v1` release](https://github.com/xinyizhanglab/perturbation-decomposition/releases/tag/rep4rev-v1).
Download and usage instructions are provided below.

Install the processed response data, DepMap features, and Hallmark gene sets.
No model is needed. To regenerate figures from saved plotting data, skip this
step and install the saved results below:

```bash
uv run python -m data.training_inputs install \
  --catalog "$DATA_DOWNLOADS/inputs.json" --bundles processed --out_dir data
uv run python -m data.artifacts validate --data_dir data/processed --dataset all
```

| Dataset | Files under `data/processed/` |
|---------|------------------------------|
| Genetic responses and DepMap | `genetic/{k562,rpe1,hepg2,jurkat}.pkl`, `genetic/depmap_embeddings.pkl` |
| Norman | `norman/norman.pkl`, `norman/depmap_embeddings.pkl`, `norman/split_info.json` |
| Tahoe | `tahoe/pseudobulk_deltas.pkl`, Morgan/CC features, and drug metadata |
| Pathways | `hallmark.gmt` |

## Pre-computed results

Regenerate supported paper figures from saved numerical results, without
inference or retraining.

| Files | Install location | Use |
|-------|------------------|-----|
| Plotting data and reference figures | `results/precomputed` | Saved numerical plotting inputs and finished figures for comparison |
| Saved predictions | `results/predictions` | Saved model predictions |
| Model weights | `checkpoints/released-models` | Weights and inputs for inference |
| Processed data | `data/processed` | Retraining and response analyses |
| Analysis data | `results/analysis-inputs` | Saved response and prediction data |

**Regenerate figures from saved results**

```bash
uv run python -m data.artifacts install \
  --index "$RESULTS_DOWNLOADS/figures-v4.tar.gz.json" \
  --archive "$RESULTS_DOWNLOADS/figures-v4.tar.gz" --out_dir results/precomputed
uv run python -m figures.render --all --mode original \
  --results_dir results/precomputed --out_dir results/figures/reproduced
```

This runs the preserved plotting code with saved numerical inputs and rendering
settings. It does not run models or repeat the underlying analyses.
`--all` regenerates only supported panels, not every paper figure.

## Main analyses

Install the saved figure inputs above first. Commands below use
`results/precomputed` by default and write to new output directories.
All redraw commands use saved numbers and preserved rendering settings; no models
or scientific analyses are rerun. Figures 1 and 4 follow the paper's panel lettering.
Other plots are identified by content and their original script names.

### Decomposition

**Figure 1 — response decomposition**

Regenerate all supported numerical panels together:

```bash
uv run python -m figures.render --mode original \
  --figure fig1_f --figure fig1_b --figure fig1_d --figure fig1_dsup --figure fig1_e \
  --out_dir results/figures/figure1
```

Or select a panel:

| Panel / plot | Command |
|-------------|---------|
| 1c — Full and template-removed decomposition | `uv run python -m figures.fig1_f --mode original --out_dir results/figures/fig1c` |
| 1d — Variance captured by the template direction | `uv run python -m figures.fig1_b --mode original --out_dir results/figures/fig1d` |
| 1e — Effective rank | `uv run python -m figures.fig1_c --mode reference --out_dir results/figures/reference/fig1e` |
| 1f — Cell-line-template Hallmark enrichment | `uv run python -m figures.fig1_d --mode original --out_dir results/figures/fig1f` |
| 1g — Template variance along baseline identity axes | `uv run python -m figures.fig1_dsup --mode original --out_dir results/figures/fig1g` |
| 1i — Control-PC capture of template and residual | `uv run python -m figures.fig1_e --mode original --out_dir results/figures/fig1i` |

Panels 1a–b are schematics, not numerical plotting entrypoints.

Optionally, compute the balanced response ANOVA,
`delta = mu + alpha_c + beta_p + gamma_{c,p}`, from the processed inputs:

```bash
uv run python -m decomposition.anova --data_dir data/processed/genetic \
  --out_dir results/decomposition/run1
```

This is not the paper's split-half noise-corrected calculation. Use the saved
figure commands for that panel. Prediction decomposition is included under Models.

### Hallmark and pathway figures

The Figure 1f command above selects the current μ + α_c Hallmark heatmap with
NES labels and FDR dots; its original script name is `fig1_d`.
The supplied gene-set definitions are in `data/processed/hallmark.gmt`.

| Panel / plot | Command |
|-------------|---------|
| Saved pathway-comparison panels | `uv run python -m figures.fig_pathway_crosscl --mode original --out_dir results/figures/pathways` |

These commands do not repeat GSEA. New pathway-prediction fits are covered under
retraining below.

### Other response figures

**Figure 2 — perturbation prediction and DepMap geometry**

```bash
uv run python -m figures.render --mode original \
  --figure fig2_a --figure fig2_b --figure fig2_c --figure fig2_d --figure fig2_e \
  --out_dir results/figures/figure2
```

| Panel / plot | Command |
|-------------|---------|
| Within-cell-line model comparison (`fig2_a`) | `uv run python -m figures.fig2_a --mode original --out_dir results/figures/fig2a` |
| DepMap pairwise similarity versus response similarity (`fig2_b`) | `uv run python -m figures.fig2_b --mode original --out_dir results/figures/fig2b` |
| Response-aligned DepMap projection (`fig2_c`) | `uv run python -m figures.fig2_c --mode original --out_dir results/figures/fig2c` |
| Current multi-model rank ablation (`fig2_d`) | `uv run python -m figures.fig2_d --mode original --out_dir results/figures/fig2d` |
| MORPH representation stages (`fig2_e`) | `uv run python -m figures.fig2_e --mode original --out_dir results/figures/fig2e` |

**Figure 3 — supported transfer plots**

```bash
uv run python -m figures.render --mode original \
  --figure fig3_a --figure fig3_c --figure fig3_d --figure fig3_e \
  --out_dir results/figures/figure3
```

| Panel / plot | Command |
|-------------|---------|
| 3a — Full transfer comparison, including BioLORD/scGen | `uv run python -m figures.fig3_a --mode original --out_dir results/figures/fig3a` |
| Unseen cell line and unseen perturbations (`fig3_c`) | `uv run python -m figures.fig3_c --mode original --out_dir results/figures/double-unseen` |
| Target response landscape (`fig3_d`) | `uv run python -m figures.fig3_d --mode original --out_dir results/figures/target-landscape` |
| Source-basis coverage versus transfer (`fig3_e`) | `uv run python -m figures.fig3_e --mode original --out_dir results/figures/basis-vs-transfer` |

**Figure 4 — prediction decomposition**

```bash
uv run python -m figures.render --mode original \
  --figure fig4_a --figure fig4_b --figure fig4_c \
  --out_dir results/figures/figure4
```

| Panel / plot | Command |
|-------------|---------|
| 4a — Within-cell-line template and residual recovery | `uv run python -m figures.fig4_a --mode original --out_dir results/figures/fig4a` |
| 4b — Independent α/β/γ recovery; MORPH controls retained, no target perturbations | `uv run python -m figures.fig4_b --mode original --out_dir results/figures/fig4b` |
| 4c — Interaction recovery before/after +30% target perturbations | `uv run python -m figures.fig4_c --mode original --out_dir results/figures/fig4c` |

Panel 4b decomposes observed and predicted tensors independently. Panel 4c compares
interaction recovery, not the older two-column performance/calibration figure.
scGen has no +30% result.

**Additional saved plots**

| Panel / plot | Command |
|-------------|---------|
| Template-removal method comparison | `uv run python -m figures.fig1_suppl_decomp --mode original --out_dir results/figures/template-removal` |
| DepMap feature/rank supplementary plots | `uv run python -m figures.fig2_supp2_2 --mode original --out_dir results/figures/depmap-supplementary` |
| Separate three-source transfer plot | `uv run python -m figures.fig3_b --mode original --out_dir results/figures/three-source` |
| DepMap versus direct source-response transfer | `uv run python -m figures.fig3_source_vs_depmap --mode original --out_dir results/figures/source-vs-depmap` |
| Norman combinations by unseen-gene count | `uv run python -m figures.fig3_morph_double_unseen_combo --mode original --out_dir results/figures/norman-combinations` |
| Cross-cell-line template/residual attribution | `uv run python -m figures.fig4_crosscl_decomp --mode original --out_dir results/figures/crosscl-template-residual` |
| Additional native performance/calibration plot | `uv run python -m figures.fig4_calibration --mode original --out_dir results/figures/native-calibration` |
| Tahoe source-basis saturation | `uv run python -m figures.fig_tahoe_b_saturation --mode original --out_dir results/figures/tahoe-source-basis` |
| Tahoe negative controls | `uv run python -m figures.fig_tahoe_c_controls --mode original --out_dir results/figures/tahoe-controls` |

`fig2_d2` is an alias for `fig2_d`; `fig3_crosscl_transfer` for `fig3_a`;
`fig4_d` for `fig4_c`. The aliases use the same saved drawings, not extra paper panels.

**Supplementary and Tahoe figures**

| Panel / plot | Command |
|-------------|---------|
| Supplementary 1b — Template/PC1 alignment | `uv run python -m figures.fig1_supp_alignment --mode original --out_dir results/figures/supp1b` |
| Supplementary 1d — Hallmark enrichment | `uv run python -m figures.fig1_supp_hallmark --mode original --out_dir results/figures/supp1d` |
| Supplementary 1e — Split-half noise | `uv run python -m figures.fig1_supp_noise --mode original --out_dir results/figures/supp1e` |
| Supplementary 1f — ANOVA diagnostics | `uv run python -m figures.fig1_supp_anova --mode original --out_dir results/figures/supp1f` |
| Supplementary 2a — Similarity deciles | `uv run python -m figures.fig2_supp_deciles --mode original --out_dir results/figures/supp2a` |
| Supplementary 2c — Pairwise scatter | `uv run python -m figures.fig2_supp_scatter --mode original --out_dir results/figures/supp2c` |
| Supplementary 3a — Ridge/MLP transfer heatmaps | `uv run python -m figures.fig3_supp1_heatmaps --mode original --out_dir results/figures/supp3a` |
| Supplementary 3b — STATE/MORPH calibration | `uv run python -m figures.fig3_supp1_slopes --mode original --out_dir results/figures/supp3b` |
| Supplementary 3c — Target calibration curves | `uv run python -m figures.fig3_supp1_calibration --mode original --out_dir results/figures/supp3c` |
| Supplementary 3d — Basis coverage and transfer | `uv run python -m figures.fig3_supp1_basis --mode original --out_dir results/figures/supp3d` |
| Supplementary 3e — Reference baselines | `uv run python -m figures.fig3_supp1_baselines --mode original --out_dir results/figures/supp3e` |
| Full Figure 1 supplementary set | `uv run python -m figures.fig1_supplementary_full --mode reference --out_dir results/figures/reference/fig1-supplementary` |
| Figure 2 supplementary set 1 | `uv run python -m figures.fig2_supp2_1 --mode reference --out_dir results/figures/reference/fig2-supplementary` |
| Figure 3 supplementary set | `uv run python -m figures.fig3_supp1 --mode original --out_dir results/figures/fig3-supplementary` |
| Figure 4 supplementary set | `uv run python -m figures.fig4_supplementary --mode original --out_dir results/figures/fig4-supplementary` |
| Full Tahoe figure | `uv run python -m figures.fig_tahoe_panels --mode reference --out_dir results/figures/reference/tahoe` |

## Models

Choose one model and one route. Use a new output directory each time.

Benchmarks cover within-cell-line CV, cross-cell-line transfer and calibration,
Norman combinations, Tahoe drugs, and pathway prediction.

| Model | 1. Saved predictions | 2. Inference | 3. Retraining |
|-------|----------------------|--------------|---------------|
| Ridge (DepMap) | Yes | Matched-transfer and new fits | Included |
| MLP (DepMap) | Yes | Matched-transfer and new fits | Included |
| MORPH | Yes | Benchmark/source checkpoints | Pinned upstream source + locked environment |
| STATE | Yes | Source-only/calibrated checkpoints | Locked native environment |
| scGPT | Yes | Fine-tuned checkpoints | Recorded driver + locked environment |
| BioLORD | Yes | Transfer/calibration checkpoints | Included |
| scGen | Yes | New fits only | Included |

**Optional model files**

Saved-figure inputs and processed response data are already in
`reproduction_files/`. Optional predictions, weights, and training inputs are in
the [`rep4rev-v1` release](https://github.com/xinyizhanglab/perturbation-decomposition/releases/tag/rep4rev-v1).
They are not needed to redraw saved figures.

For the download commands below, use Bash and the GitHub CLI (`gh`).
Sign in with `gh auth login` if needed. Run this once in the same shell:

```bash
download() (
  set -eo pipefail
  ARCHIVE="$1.tar.gz"
  gh release download rep4rev-v1 \
    --repo xinyizhanglab/perturbation-decomposition \
    --pattern "$ARCHIVE*" --dir reproduction_files --skip-existing
  if [ ! -f "reproduction_files/$ARCHIVE" ]; then
    cat reproduction_files/"$ARCHIVE".part-??? > "reproduction_files/$ARCHIVE.partial"
    mv -n "reproduction_files/$ARCHIVE.partial" "reproduction_files/$ARCHIVE"
  fi
)
```

This downloads the archive and its index to `reproduction_files/`, joining split
files automatically. Existing downloads are reused. Installation checks the
archive checksum. On the cluster, run large downloads and installations inside
a compute allocation, not on a login node.

**1. Saved predictions**

Regenerate supported model-comparison figures using the saved numerical
results above. To work directly with saved predictions, choose a model:

```bash
download predictions-v1
uv run python -m data.artifacts install \
  --index "$RESULTS_DOWNLOADS/predictions-v1.tar.gz.json" \
  --archive "$RESULTS_DOWNLOADS/predictions-v1.tar.gz" --out_dir results/predictions
RUN=results/reproduce/saved
uv run python -m models.run --model ridge --target all --saved --out_dir "$RUN"
```

Replace `ridge` with `mlp`, `morph`, `state`, `scgpt`, `biolord`, or `scgen`.
Use `--target k562`, `rpe1`, `hepg2`, or `jurkat` for one target.
Outputs go to `$RUN/<target>`. No model SDK is needed.
scGen selects the complete saved binary-fit cohort.

Where weights are available, inference uses the same experiment as saved predictions:

| Model | Default experiment |
|-------|--------------------|
| Ridge / MLP | Three-source transfer, source-only |
| MORPH / scGPT | Within-cell-line prediction |
| STATE | Source-only transfer |
| BioLORD / scGen | Three-source transfer |

The default seed/fold is 0; use `--repeat` to select another released repeat.
The STATE source-only example has repeat 0 only.
Continue with “Plot model outputs” below.

**2. Inference**

Install the model weights:

```bash
download models-v1
uv run python -m data.artifacts install \
  --index "$RESULTS_DOWNLOADS/models-v1.tar.gz.json" \
  --archive "$RESULTS_DOWNLOADS/models-v1.tar.gz" --out_dir checkpoints/released-models
```

Choose the model exactly as above. The command selects its prepared environment
automatically; install that environment under External model setup first.

```bash
RUN=results/reproduce/inference
uv run python -m models.run --model ridge --target all \
  --out_dir "$RUN" --trust_checkpoint --execute
```

Omit `--execute` to inspect the selected experiment and environment without
running inference. For native models, `--device cpu` or `--device cuda:0` overrides
automatic device selection. Existing weights and matching inputs are selected
together; no model is retrained.

For a new scGen binary fit, provide its checkpoint and matching input request:

```bash
uv run python -m models.run --model scgen --target k562 \
  --checkpoint /path/to/new-binary-fit/checkpoint --inputs /path/to/binary-request.npz \
  --out_dir results/reproduce/scgen-new --trust_checkpoint --execute
```

That fit supports only its recorded target and perturbation.

**3. Retraining**

Ridge/MLP use the processed inputs under Data; native inputs are listed later
in this route. For DepMap Ridge/MLP:

```bash
uv run --extra analysis --extra gpu python -m benchmarks.cross_cl \
  --model all --target all --n_sources 3 --device cuda:0 \
  --data_dir data/processed/genetic \
  --depmap data/processed/genetic/depmap_embeddings.pkl \
  --out_dir results/train/depmap
```

Both models save weights. For a Ridge-only CPU run, use `--model ridge --device cpu`
without the GPU extra. Generate a plot-ready panel from the new Ridge fits:

```bash
RUN=results/reproduce/retrained
for CELL in k562 rpe1 hepg2 jurkat; do
  FIT=$(find results/train/depmap/ridge -maxdepth 1 -type d -name "*_to_${CELL}")
  uv run python -m data.inference --dataset "data/processed/genetic/$CELL.pkl" \
    --depmap data/processed/genetic/depmap_embeddings.pkl \
    --perturbations "$FIT/predictions.npz" --out_dir "$RUN/inputs/$CELL" --execute
  uv run python -m models.predict --model ridge --checkpoint "$FIT/checkpoint" \
    --inputs "$RUN/inputs/$CELL/inputs.npz" --target "$CELL" --device cpu \
    --out_dir "$RUN/$CELL" --execute
done
```

Other baseline analyses:

```bash
uv run --extra analysis python -m benchmarks.within_cl --model ridge --cell_line k562 \
  --data_dir data/processed/genetic --depmap data/processed/genetic/depmap_embeddings.pkl \
  --out_dir results/train/within --device cpu
uv run --extra analysis --extra gpu python -m benchmarks.norman_doubles \
  --data_dir data/processed/norman --out_dir results/train/norman --device cuda:0
uv run --extra analysis python -m benchmarks.tahoe_drugs \
  --data_dir data/processed/tahoe --features all --out_dir results/train/tahoe
uv run --extra analysis python -m benchmarks.pathway_prediction --setting cross_cl \
  --data_dir data/processed/genetic --depmap data/processed/genetic/depmap_embeddings.pkl \
  --gmt data/processed/hallmark.gmt --out_dir results/train/pathway
```

Tahoe also accepts `--features source --rank_sweep` or `--moa_holdout`.
For new Norman/Tahoe checkpoints, add `--save_checkpoints --gene_names /path/to/ordered-genes.json`;
pathway checkpoint export uses `--save_checkpoints`.

**Native training inputs**

Download only the inputs for your model, using `download` from above:

| Model | Download | Install selection |
|-------|----------|-------------------|
| STATE | `download inputs-genetic-v1` | `--model state` |
| MORPH | `download inputs-morph-v1` | `--model morph` |
| scGPT (K562) | `download inputs-scgpt-k562-v1` and `download inputs-scgpt-pretrained-v1` | `--model scgpt --target k562` |
| BioLORD | `download inputs-biolord-v1` and `download inputs-references-v1` | `--model biolord` |
| scGen | `download inputs-genetic-v1` and `download inputs-references-v1` | `--model scgen` |
| BioLORD calibration | `download inputs-calibration-v1` and `download inputs-references-v1` | `--bundles calibration` |

For example, download and install BioLORD's inputs:

```bash
download inputs-biolord-v1
download inputs-references-v1
uv run python -m data.training_inputs install \
  --catalog "$DATA_DOWNLOADS/inputs.json" --model biolord --out_dir data
```

For another model, use its downloads and install selection from the table.
For scGPT, replace `k562` with `rpe1`, `hepg2`, or `jurkat` in both places.
For all four targets, download all four cell-line files and omit `--target`;
download the pretrained file only once. Ridge/MLP use the processed inputs
already installed under Data.

Inputs are installed under `data/inputs/`. Required companions are selected
automatically; Python environments are not installed. Matching installed files
are checked and reused; conflicting files are not overwritten.

Training inputs retain the recorded matrices, labels, memberships, and splits.
scGPT includes its stored cell/control pairings; BioLORD/calibration include
the prepared inputs and required reference data.

For native models, prepare their environment below. These commands print plans;
add `--execute` to run inside a compute allocation.

```bash
REFERENCES=data/inputs/references
```

BioLORD:

```bash
uv run --no-sync --project models/biolord python -m models.biolord.train --stage prepare \
  --prepared_inputs data/inputs/biolord --reference_dir "$REFERENCES" \
  --out_dir results/train/native
uv run --no-sync --project models/biolord python -m models.biolord.train --stage train --task 0 \
  --prepared_inputs data/inputs/biolord --reference_dir "$REFERENCES" --out_dir results/train/native
uv run --no-sync --project models/biolord python -m models.biolord.train --stage report \
  --prepared_inputs data/inputs/biolord --reference_dir "$REFERENCES" --out_dir results/train/native
```

The prepare stage copies verified frozen inputs without resampling.
Run tasks 0–19 before the report; it writes the transfer results and a new transfer
figure. Calibration uses `calibration-prepare`, `calibration-train` (tasks 0–39),
and `calibration-report`, with `--prepared_inputs data/inputs/calibration`.

scGen:

```bash
uv run --no-sync --project models/scgen python -m models.scgen.train --stage prepare \
  --data_dir data/inputs/genetic --reference_dir "$REFERENCES" --out_dir results/train/native
uv run --no-sync --project models/scgen python -m models.scgen.train --stage train \
  --shard 0 --workers 64 --save_checkpoints --checkpoint_tasks 0 \
  --data_dir data/inputs/genetic --reference_dir "$REFERENCES" --out_dir results/train/native
```

Run shards 0–63 for the full experiment; the final shard writes its report and
transfer figure. `--checkpoint_tasks` chooses weights to retain, not fits to train.
Omit that filter to save all new binary weights.

MORPH, STATE, and scGPT use the locked environments under External model setup.
For MORPH, fetch the recorded upstream revision once:

```bash
git clone --no-checkout https://github.com/uhlerlab/MORPH.git external/MORPH
git -C external/MORPH checkout --detach 1ad06d4591434cc4ab03c12c461fe0d80b087b11
```

Then use the installed inputs:

```bash
uv run python -m models.morph.train --data_dir data/inputs/morph \
  --config configs/training/morph.json --out_dir results/train/morph
uv run python -m models.state.train --data_dir data/inputs/genetic \
  --config configs/training/state.json --out_dir results/train/state
uv run python -m models.scgpt.train --data_dir data/inputs/scgpt \
  --config data/inputs/scgpt/replogle_k562_essential/training.json \
  --out_dir results/train/scgpt
```

These commands print plans. Add `--check` to verify native imports and, for STATE,
its Hydra configuration without loading matrices or training. Add `--execute`
to train in a compute allocation.
Each launcher selects its prepared environment automatically. MORPH
checks its pinned source files and applies its compatibility fix only to the
isolated training copy. STATE uses the same GPT2 helper as inference.
scGPT uses the supplied driver, pretrained weights, and cached graphs.
Source inputs are not modified; weights and native outputs stay under `--out_dir`.

**Plot model outputs — after the selected route**

For compatible cross-cell-line outputs, keep `RUN` from the route above:

```bash
cat > "$RUN/panel.json" <<'JSON'
{
  "contexts": ["k562", "rpe1", "hepg2", "jurkat"],
  "predictions": {
    "k562": "k562/predictions.npz",
    "rpe1": "rpe1/predictions.npz",
    "hepg2": "hepg2/predictions.npz",
    "jurkat": "jurkat/predictions.npz"
  }
}
JSON
uv run python -m decomposition.run --manifest "$RUN/panel.json" \
  --out_dir "$RUN/components" --execute
uv run python -m figures.components --results_dir "$RUN/components" \
  --out_dir "$RUN/figures"
```

Prediction files must have verified, matching gene order and shared perturbations; mismatched
native gene axes are rejected. These are **new component plots, not replacements
for the archived paper panels**. Fresh inference/retraining does not automatically
rebuild every original paper panel.

### External model setup

Saved predictions and portable Ridge/MLP inference need only the core environment.
MLP training uses the core GPU extra; native models have their own environments:

```bash
uv sync --locked --extra analysis --extra gpu
uv sync --locked --project models/morph
uv sync --locked --project models/biolord
uv sync --locked --project models/scgen
uv sync --locked --project models/state
uv sync --locked --project models/scgpt
```

Install only the environments you use. MORPH, STATE, and scGPT environments include
their training dependencies as well as inference support:

| Model | Native source / runtime |
|-------|-------------------------|
| MORPH | Commit `1ad06d4591434cc4ab03c12c461fe0d80b087b11`; Python 3.10, PyTorch 2.10.0+cu128 |
| STATE | `arc-state==0.10.2`; Python 3.11, PyTorch 2.10.0+cu128 |
| scGPT | `scgpt==0.2.4`, `cell-gears==0.0.2`; Python 3.10, PyTorch 2.10.0+cu128 |

The lockfiles pin all remaining dependencies. No Conda environment or separate
training checkout is needed for STATE/scGPT.

## Metrics
| Metric | Description | Primary |
|--------|-------------|---------|
| pref | Template-removed Pearson: corr(delta - T_train, delta_hat - T_train) | Yes |
| std_r | Standard Pearson: corr(delta, delta_hat) | No |

## Repository structure

```text
perturbation-decomposition/
├── configs/datasets.yaml          # Dataset metadata and reference defaults
├── data/
│   ├── process.py                 # Raw h5ad to processed responses
│   ├── prepare_release.py         # Export existing research inputs
│   ├── artifacts.py               # Install and verify downloaded files
│   ├── training_inputs.py         # Model-specific preprocessed inputs
│   ├── check.py                   # Environment/configuration checks
│   └── processed/                 # Downloaded processed inputs
├── decomposition/
│   ├── anova.py                   # Existing response decomposition
│   └── run.py                     # Labeled response/prediction panel interface
├── models/
│   ├── run.py                     # Select predictions or weights by model and target
│   ├── ridge.py                   # Ridge and MLP baselines
│   ├── morph/
│   ├── state/
│   ├── scgpt/
│   ├── biolord/                   # BioLORD commands and uv environment
│   └── scgen/                     # scGen commands and uv environment
├── benchmarks/                    # Dataset and transfer benchmarks
├── evaluation/metrics.py           # Prediction evaluation
├── figures/                       # Paper figure entrypoints and style
├── reproduction/                  # Shared file and plotting utilities
├── scripts/                       # Workflow dispatch
├── slurm/                         # Optional cluster job templates
├── tests/                         # Small CPU-only tests
└── results/                       # Downloaded predictions and generated outputs
```

Run the software tests with `uv run python -m unittest discover -s tests -v`.

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

MIT. Bundled third-party code retains its accompanying license.
