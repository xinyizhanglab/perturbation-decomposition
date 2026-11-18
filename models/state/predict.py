"""state checkpoint inference and saved-prediction access."""
import argparse, pickle, logging
from pathlib import Path
import numpy as np

from evaluation.metrics import pref, std_r, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def main_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--state_dir", type=Path, default=Path("external/STATE"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    log.info(f"=== STATE training for {args.dataset} ===")
    log.info("\nSTATE training requires the original STATE pipeline.")
    log.info("See README.md for setup instructions.")
    log.info(f"\nOnce trained, run evaluation:")
    log.info(f"  python -m models.state.predict \\")
    log.info(f"    --pred_dir {args.out_dir} \\")
    log.info(f"    --data_dir {args.data_dir}")


def main_predict():
    """Load STATE predictions and evaluate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, required=True,
                        help="Directory with seed*/predictions.npz")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.pred_dir

    log.info(f"=== Evaluating STATE predictions ===")

    # Load all seeds
    all_prefs, all_stdr = [], []
    for seed_dir in sorted(args.pred_dir.glob("seed*")):
        f = seed_dir / "predictions.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        yt, yp, T = d["true_deltas"], d["predictions"], d["train_mean"]
        pr = pref(yt, yp, T)
        sr = std_r(yt, yp)
        all_prefs.append(pr)
        all_stdr.append(sr)
        log.info(f"  {seed_dir.name}: n={len(yt)}, pref={pr:.4f}, std_r={sr:.4f}")

    if all_prefs:
        log.info(f"\n  Mean pref: {np.mean(all_prefs):.4f} ± {np.std(all_prefs):.4f}")
        log.info(f"  Mean std_r: {np.mean(all_stdr):.4f} ± {np.std(all_stdr):.4f}")


if __name__ == "__main__":
    import sys
    if any(arg.split("=", 1)[0] == "--pred_dir" for arg in sys.argv[1:]):
        main_predict()
    else:
        from models.predict import main as predict_main
        predict_main(["--model", "state", *sys.argv[1:]])
