# MORPH wrapper

## Setup

1. Clone the MORPH repository:
```bash
git clone https://github.com/uhlerlab/MORPH external/MORPH
```

2. Register your dataset in MORPH's data paths:
```bash
# Add to external/MORPH/data/scdata_file_path.csv:
dataset_name,/path/to/your/data.h5ad

# Add to external/MORPH/data/perturb_embed_file_path.csv:
DepMap_GeneEffect,dataset_name,/path/to/depmap_embeddings.pkl
```
**Column order matters**: representation_type, dataset_name, file_path

3. Prepare data:
The h5ad must have:
- `obs['gene']`: perturbation labels ("GENE" for singles, "GENE1+GENE2" for doubles, "non-targeting" for controls)
- `.X`: HVG expression matrix (not in obsm)

## Training

```bash
cd external/MORPH/morph
python run.py \
    --dataset_name your_dataset \
    --leave_out_test_set_id benchmark \
    --device cuda:0 \
    --model MORPH \
    --label DepMap_GeneEffect \
    --epochs 100 \
    --batch_size 32
```

## Key architecture details

MORPH composes double-gene perturbations by summing the two perturbation embeddings in latent space after the perturbation encoder:
```python
z_ptb_ft = z_ptb_1 + z_ptb_2  # additive composition
```

The perturbation encoder maps DepMap features through a learned MLP before this summation. Our stage ablation shows this encoder is monotonically destructive: raw DepMap → pert encoder → pre-attention → post-attention, each stage loses information.
