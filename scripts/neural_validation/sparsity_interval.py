"""Paired uncertainty in the Electronics + ID sparsity experiment."""
import argparse
import json
from pathlib import Path

import numpy as np


def scores(root, method, fraction, seed):
    if fraction == 1:
        name = f"shared-id-1.0-{seed}" if method == "shared" else f"buy-{seed}"
    else:
        name = f"{method}-{fraction}-{seed}"
    records = json.loads((root / name / "validation_per_query.json").read_text())
    return {(row["source"], row["category"]): row["ndcg10"] for row in records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_runs", type=Path, help="Replay output for electronics-id-v2")
    root = parser.parse_args().dataset_runs
    endpoints = []
    keys = None
    for fraction in (1, 0.25):
        paired = []
        for seed in (42, 43, 44):
            shared = scores(root, "shared", fraction, seed)
            buy = scores(root, "buy", fraction, seed)
            if keys is None:
                keys = sorted(shared)
            if sorted(shared) != keys or sorted(buy) != keys:
                raise ValueError("Evaluation query sets differ")
            paired.append([shared[key] - buy[key] for key in keys])
        endpoints.append(np.mean(paired, axis=0))
    delta = endpoints[1] - endpoints[0]
    _, groups = np.unique([source for source, _ in keys], return_inverse=True)
    totals = np.bincount(groups, weights=delta)
    counts = np.bincount(groups)
    rng = np.random.default_rng(20261120)
    draws = rng.integers(len(counts), size=(5000, len(counts)))
    means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    print("Difference:", float(delta.mean()))
    print("95% source-cluster bootstrap interval:", np.quantile(means, [0.025, 0.975]))


if __name__ == "__main__":
    main()
