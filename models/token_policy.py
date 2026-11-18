"""Explicit reproduction of scGPT's recorded vocabulary-default policy."""

import hashlib
import json

from reproduction.io import read_json, sha256


def axis_sha256(names):
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def recorded_policy(vocabulary_path, names):
    vocabulary = read_json(vocabulary_path)
    if "<pad>" not in vocabulary:
        raise ValueError("The recorded scGPT vocabulary has no <pad> token.")
    return {
        "schema_version": 1, "policy": "scgpt_0.2.4_GeneVocab_default",
        "default_token": "<pad>", "default_index": vocabulary["<pad>"],
        "vocabulary_sha256": sha256(vocabulary_path), "gene_axis_sha256": axis_sha256(names),
        "unknown_genes": sorted(set(names) - set(vocabulary)),
        "protocol": "Original GeneVocab.from_file defaults to <pad>; all output genes are retained.",
    }


def validate_policy(path, vocabulary_path, names):
    actual = read_json(path)
    expected = recorded_policy(vocabulary_path, names)
    for key in ("schema_version", "policy", "default_token", "default_index",
                "vocabulary_sha256", "gene_axis_sha256", "unknown_genes"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Recorded scGPT token policy differs at {key}; no genes are silently dropped.")
    return actual
