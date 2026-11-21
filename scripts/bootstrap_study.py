"""Compute cluster-bootstrap intervals for paired test-query metric differences."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def interval(cluster_totals: np.ndarray, cluster_queries: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(2026)
    estimates = []
    for _ in range(20):
        index = rng.integers(len(cluster_queries), size=(100, len(cluster_queries)))
        estimates.extend(cluster_totals[index].sum(axis=1) / cluster_queries[index].sum(axis=1))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def bootstrap(study: Path, personalized: bool) -> None:
    comparisons = pd.read_csv(study / "paired_deltas.csv")
    keys = ["source_id", "target_category", "user_id"]
    cluster = "user_id" if personalized else "source_id"
    metrics = ["recall@10", "ndcg@10", "hit_rate@10"]
    estimates = []
    for fraction, group in comparisons.groupby("buy_fraction"):
        names = group[["model", "reference"]].drop_duplicates().itertuples(index=False, name=None)
        seeds = sorted(group["seed"].unique())
        runs = []
        for seed in seeds:
            path = study / f"test-{int(100 * fraction)}-seed-{seed}" / "recommendations.csv.gz"
            columns = set(keys + ["model", *metrics])
            predictions = pd.read_csv(path, keep_default_na=False, dtype={key: str for key in keys},
                                      usecols=lambda name: name in columns)
            if "hit_rate@10" not in predictions:
                predictions["hit_rate@10"] = predictions["recall@10"].gt(0).astype("uint8")
            runs.append(predictions.assign(seed=seed))
        values = pd.concat(runs, ignore_index=True).groupby(["model", *keys], sort=False)[metrics].mean()
        for model, reference in names:
            difference = values.loc[model] - values.loc[reference]
            if difference.isna().any().any():
                raise ValueError(f"Queries differ for {model} and {reference}")
            clustered = difference.groupby(cluster)
            totals = clustered.sum()
            counts = clustered.size().to_numpy()
            for metric in metrics:
                low, high = interval(totals[metric].to_numpy(), counts)
                estimates.append({
                    "buy_fraction": fraction, "model": model, "reference": reference,
                    "metric": metric, "mean_delta": difference[metric].mean(),
                    "ci_2.5": low, "ci_97.5": high, "clusters": len(counts),
                    "cluster_key": cluster, "bootstrap_draws": 2000,
                })
    pd.DataFrame(estimates).to_csv(study / "paired_bootstrap.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--personalized", action="store_true")
    args = parser.parse_args()
    bootstrap(args.study, args.personalized)
