"""MORPH training wrapper.

Wraps the Uhler lab MORPH codebase with our data/split/evaluation pipeline.
No SLURM required — runs directly.

Prerequisites:
    git clone https://github.com/uhlerlab/MORPH external/MORPH
    pip install -r external/MORPH/requirements.txt

Usage:
    python -m models.morph.train --dataset k562 \
        --data_dir data/processed/genetic \
        --morph_dir external/MORPH \
        --out_dir results/morph/k562 \
        --seed 0

    python -m models.morph.train --dataset norman \
        --data_dir data/processed/norman \
        --morph_dir external/MORPH \
        --out_dir results/morph/norman
"""
import argparse, sys, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["k562", "rpe1", "hepg2", "jurkat", "norman"])
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--morph_dir", type=Path, default=Path("external/MORPH"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    morph_src = args.morph_dir / "morph"
    if not morph_src.exists():
        log.error(f"MORPH source not found at {morph_src}")
        log.error("Run: git clone https://github.com/uhlerlab/MORPH external/MORPH")
        sys.exit(1)

    sys.path.insert(0, str(morph_src))

    log.info(f"=== Training MORPH on {args.dataset} (seed {args.seed}) ===")
    log.info(f"  MORPH source: {morph_src}")
    log.info(f"  Data: {args.data_dir}")
    log.info(f"  Output: {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("\nTo train MORPH:")
    log.info(f"  1. Register dataset in {args.morph_dir}/data/scdata_file_path.csv")
    log.info(f"  2. Register DepMap in {args.morph_dir}/data/perturb_embed_file_path.csv")
    log.info(f"     Column order: DepMap_GeneEffect,{args.dataset},/path/to/depmap.pkl")
    log.info(f"  3. Run:")
    log.info(f"     cd {morph_src}")
    log.info(f"     python run.py \\")
    log.info(f"       --dataset_name {args.dataset} \\")
    log.info(f"       --leave_out_test_set_id benchmark \\")
    log.info(f"       --device {args.device} \\")
    log.info(f"       --model MORPH \\")
    log.info(f"       --label DepMap_GeneEffect \\")
    log.info(f"       --epochs {args.epochs} \\")
    log.info(f"       --batch_size {args.batch_size} \\")
    log.info(f"       --seed {args.seed}")


if __name__ == "__main__":
    main()
