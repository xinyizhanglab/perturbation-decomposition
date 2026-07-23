"""scGPT perturbation prediction evaluation.

scGPT uses pretrained gene embeddings + per-gene perturbation flags.
Our analysis showed the conditioning interface is near-diagonal
(<1% cross-gene coupling).

Prerequisites:
    pip install scgpt
    Download checkpoint: scGPT_human

Usage:
    python -m models.scgpt.predict \
        --pred_dir results/scgpt/k562 \
        --data_dir data/processed/genetic \
        --dataset k562
"""
import argparse, pickle, logging
from pathlib import Path
import numpy as np

from evaluation.metrics import pref, std_r, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def evaluate_scgpt_predictions(pred_dir, data_dir, dataset):
    """Evaluate precomputed scGPT predictions."""
    pred_dir = Path(pred_dir)
    data_dir = Path(data_dir)

    # Load data for ground truth
    with open(data_dir / f"{dataset}.pkl", "rb") as f:
        data = pickle.load(f)

    # Load scGPT predictions
    results = []
    for f in sorted(pred_dir.glob("**/predictions.npz")):
        d = np.load(f, allow_pickle=True)
        yt = d["true_deltas"]
        yp = d["predictions"]
        T = d["train_mean"]

        pr = pref(yt, yp, T)
        sr = std_r(yt, yp)

        log.info(f"  {f.parent.name}: n={len(yt)}, pref={pr:.4f}, std_r={sr:.4f}")
        results.append({"path": str(f), "pref": pr, "std_r": sr, "n": len(yt)})

    if results:
        log.info(f"\n  Mean pref: {np.mean([r['pref'] for r in results]):.4f}")

    return results




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/genetic"))
    parser.add_argument("--dataset", default="k562")
    parser.add_argument("--ablation_dir", type=Path, default=None,
                        help="Path to global-embedding-swap predictions for ablation")
    args = parser.parse_args()

    log.info(f"=== scGPT evaluation: {args.dataset} ===")
    results = evaluate_scgpt_predictions(args.pred_dir, args.data_dir, args.dataset)

    if args.ablation_dir:
        interface_ablation(args.pred_dir, args.ablation_dir, args.data_dir, args.dataset)


if __name__ == "__main__":
    main()
