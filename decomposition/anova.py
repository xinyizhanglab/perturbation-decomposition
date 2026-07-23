"""Four-component ANOVA decomposition of perturbation responses.

Decomposes δ_{c,p} = μ + α_c + β_p + γ_{c,p}

  μ:     global mean effect
  α_c:   cell-line-specific baseline (template)
  β_p:   conserved perturbation effect (shared across CLs)
  γ_{c,p}: cell-line × perturbation interaction (CL-specific residual)

Usage:
    python -m decomposition.anova --data_dir ./data/processed
"""
import argparse, pickle, logging, json
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

CLS = ["k562", "rpe1", "hepg2", "jurkat"]


def compute_anova(cl_deltas):
    """Compute 4-component ANOVA on shared perturbations.

    Args:
        cl_deltas: dict[str, dict[str, np.ndarray]]
            cl_deltas[cell_line][perturbation] = delta vector

    Returns:
        mu: (G,) global mean
        alpha: dict[str, (G,)] cell-line effects
        beta: dict[str, (G,)] conserved perturbation effects
        gamma: dict[str, dict[str, (G,)]] interaction effects
        shared_perts: list of shared perturbation names
    """
    # Find perturbations shared across all cell lines
    shared = sorted(set.intersection(*[set(cl_deltas[cl].keys()) for cl in cl_deltas]))
    cls = sorted(cl_deltas.keys())
    log.info(f"  Shared perturbations: {len(shared)}")

    # Build tensor (n_cls, n_perts, n_genes)
    D = np.array([[cl_deltas[cl][p] for p in shared] for cl in cls], dtype=np.float32)

    mu = D.mean(axis=(0, 1))                        # (G,)
    alpha_arr = D.mean(axis=1) - mu                  # (n_cls, G)
    beta_arr = D.mean(axis=0) - mu                   # (n_perts, G)
    gamma_arr = D - mu - alpha_arr[:, None, :] - beta_arr[None, :, :]  # (n_cls, n_perts, G)

    # Variance fractions
    total_var = np.var(D)
    alpha_var = np.var(alpha_arr[:, None, :] * np.ones((1, len(shared), 1)))
    beta_var = np.var(beta_arr[None, :, :] * np.ones((len(cls), 1, 1)))
    gamma_var = np.var(gamma_arr)

    log.info(f"  Variance fractions:")
    log.info(f"    α (cell line):    {alpha_var / total_var:.1%}")
    log.info(f"    β (conserved):    {beta_var / total_var:.1%}")
    log.info(f"    γ (interaction):  {gamma_var / total_var:.1%}")

    # Convert to dicts
    alpha = {cl: alpha_arr[i] for i, cl in enumerate(cls)}
    beta = {p: beta_arr[i] for i, p in enumerate(shared)}
    gamma = {cl: {p: gamma_arr[i, j] for j, p in enumerate(shared)}
             for i, cl in enumerate(cls)}

    # Transferable fraction per perturbation
    beta_frac = {}
    for j, p in enumerate(shared):
        b_var = np.var(beta_arr[j])
        g_var = np.mean([np.var(gamma_arr[i, j]) for i in range(len(cls))])
        beta_frac[p] = float(b_var / (b_var + g_var + 1e-30))

    return {
        "mu": mu, "alpha": alpha, "beta": beta, "gamma": gamma,
        "shared_perts": shared, "cell_lines": cls,
        "beta_fraction": beta_frac,
        "variance_fractions": {
            "alpha": float(alpha_var / total_var),
            "beta": float(beta_var / total_var),
            "gamma": float(gamma_var / total_var),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/processed")
    parser.add_argument("--out_dir", default="./results/decomposition")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cl_deltas = {}
    for cl in CLS:
        pkl = data_dir / f"{cl}.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                cl_deltas[cl] = pickle.load(f)["deltas"]
            log.info(f"  Loaded {cl}: {len(cl_deltas[cl])} perts")

    if len(cl_deltas) < 2:
        log.error("Need at least 2 cell lines for ANOVA")
        return

    result = compute_anova(cl_deltas)

    with open(out_dir / "anova_decomposition.pkl", "wb") as f:
        pickle.dump(result, f)

    # Save summary
    summary = {
        "n_shared": len(result["shared_perts"]),
        "cell_lines": result["cell_lines"],
        "variance_fractions": result["variance_fractions"],
        "mean_beta_fraction": float(np.mean(list(result["beta_fraction"].values()))),
    }
    with open(out_dir / "anova_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
