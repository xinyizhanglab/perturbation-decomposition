"""Export a checkpoint's explicit one-hot ordering as a prepared JSON axis."""

import argparse
from pathlib import Path

import numpy as np

from reproduction.io import sha256, write_json


def ordered_labels(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Expected a recorded label-to-one-hot-vector map")
    labels = [None]*len(mapping)
    for name, tensor in mapping.items():
        values = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if values.shape != (len(mapping),) or not np.isfinite(values).all():
            raise ValueError("Invalid one-hot map dimensions")
        index = int(values.argmax())
        expected = np.zeros(len(mapping), dtype=values.dtype)
        expected[index] = 1
        if not np.array_equal(values, expected) or labels[index] is not None:
            raise ValueError("Map must encode distinct basis vectors")
        labels[index] = str(name)
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate label identities")
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trust_checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if not args.trust_checkpoint:
        parser.error("Loading native Torch maps requires --trust_checkpoint")
    from benchmarks._runtime import require_compute
    require_compute("Native perturbation-map export")
    import torch
    labels = ordered_labels(torch.load(args.map, map_location="cpu", weights_only=False))
    write_json(args.out, labels)
    write_json(args.out.with_suffix(".provenance.json"), {"source_sha256": sha256(args.map)})


if __name__ == "__main__":
    main()
