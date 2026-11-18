"""morph checkpoint inference and saved-prediction access."""
import argparse, sys, pickle, logging
from pathlib import Path
import numpy as np

from evaluation.metrics import pref, std_r, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def predict_morph(model, ctrl_X, test_perts, depmap, ctrl_mean, device, seed=0,
                  n_samples=100):
    """Run MORPH inference: sample control cells, predict with perturbation embedding."""
    import torch
    torch.manual_seed(seed)

    preds = {}
    model.eval()

    for pn in test_perts:
        emb = depmap.get(pn, depmap.get(pn + "+ctrl", None))
        if emb is None:
            continue

        # Sample control cells
        idx = np.random.choice(len(ctrl_X), size=n_samples, replace=True)
        x_ctrl = torch.tensor(ctrl_X[idx], dtype=torch.float32, device=device)
        z_ptb = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0).expand(n_samples, -1)

        with torch.no_grad():
            x_pred = model(x_ctrl, z_ptb)
            if isinstance(x_pred, tuple):
                x_pred = x_pred[0]

        pred_mean = x_pred.cpu().numpy().mean(0)
        preds[pn] = pred_mean - ctrl_mean

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--morph_dir", type=Path, default=Path("external/MORPH"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.data_dir / f"{args.dataset}.pkl", "rb") as f:
        data = pickle.load(f)
    with open(args.data_dir / "depmap_embeddings.pkl", "rb") as f:
        depmap = pickle.load(f)

    # Load MORPH model
    sys.path.insert(0, str(args.morph_dir / "morph"))
    import torch

    log.info(f"Loading checkpoint: {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location=args.device)
    model.eval()

    # Run inference
    test_perts = sorted(data["deltas"].keys())
    ctrl_mean = data["ctrl_mean"]

    log.info(f"Running inference on {len(test_perts)} perturbations...")
    preds = predict_morph(model, data.get("ctrl_X", np.zeros((100, data["n_hvgs"]))),
                          test_perts, depmap, ctrl_mean, args.device, args.seed)

    # Save
    ok_perts = sorted(p for p in test_perts if p in preds)
    yt = np.array([data["deltas"][p] for p in ok_perts])
    yp = np.array([preds[p] for p in ok_perts])
    T = yt.mean(0)

    save_predictions(args.out_dir / "predictions.npz", ok_perts, yp, yt, T)

    pr = pref(yt, yp, T)
    sr = std_r(yt, yp)
    log.info(f"  Results: n={len(ok_perts)}, pref={pr:.4f}, std_r={sr:.4f}")
    log.info(f"  Saved to {args.out_dir}")


if __name__ == "__main__":
    import sys
    from models.predict import main as predict_main
    predict_main(["--model", "morph", *sys.argv[1:]])
