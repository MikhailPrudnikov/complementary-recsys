"""Add HitRate@10 to completed rankings and study tables without training."""

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC = "hit_rate@10"
METRIC_ORDER = [METRIC, "ndcg@10", "recall@10"]
MEAN_ORDER = ["hit_rate_mean", "ndcg_mean", "recall_mean"]


def write_csv(frame: pd.DataFrame, output: Path) -> None:
    temporary = output.with_name(output.name + ".part")
    compression = {"method": "gzip", "compresslevel": 3} if output.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(output)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, **kwargs)


def backfill(study: Path) -> None:
    test_runs = read_csv(study / "test_runs.csv")
    summary = read_csv(study / "test_summary.csv")
    rankings = sorted(study.glob("*/recommendations.csv.gz"))
    if not rankings:
        raise ValueError(f"No saved recommendations under {study}")
    scores = {}
    for path in rankings:
        rows = read_csv(path, dtype={"source_id": str, "user_id": str, "target_category": str})
        if not rows["recall@10"].between(0, 1).all():
            raise ValueError(f"Invalid Recall@10 in {path}")
        # Every evaluated query has at least one relevant target. Positive
        # recall is therefore equivalent to at least one hit in its top 10.
        hits = rows["recall@10"].gt(0).astype("uint8")
        if METRIC not in rows:
            rows[METRIC] = hits
            write_csv(rows, path)
        elif not rows[METRIC].eq(hits).all():
            raise ValueError(f"Saved HitRate disagrees with Recall in {path}")
        means = hits.groupby(rows["model"], sort=False).mean()
        metrics_path = path.parent / "metrics.csv"
        metrics = read_csv(metrics_path)
        metrics[METRIC] = metrics["model"].map(means)
        if metrics[METRIC].isna().any():
            raise ValueError(f"Missing rankings for a model in {metrics_path}")
        write_csv(metrics.sort_values(METRIC_ORDER, ascending=False), metrics_path)
        scores[path.parent.name] = means

    test_runs[METRIC] = [
        scores[f"test-{round(fraction * 100)}-seed-{seed}"].at[model]
        for fraction, seed, model in test_runs[["buy_fraction", "seed", "model"]].itertuples(index=False, name=None)
    ]
    write_csv(test_runs.sort_values(METRIC_ORDER, ascending=False), study / "test_runs.csv")
    hit_stats = test_runs.groupby(["buy_fraction", "model"])[METRIC].agg(["mean", "std"])
    for stat in ("mean", "std"):
        summary[f"hit_rate_{stat}"] = [
            hit_stats.at[(fraction, model), stat]
            for fraction, model in summary[["buy_fraction", "model"]].itertuples(index=False, name=None)
        ]
    write_csv(summary.sort_values(MEAN_ORDER, ascending=False), study / "test_summary.csv")

    deltas_path = study / "paired_deltas.csv"
    if deltas_path.exists():
        deltas = read_csv(deltas_path)
        hits_by_run = test_runs.set_index(["buy_fraction", "seed", "model"])[METRIC]
        deltas[METRIC] = [
            hits_by_run.at[(fraction, seed, model)] - hits_by_run.at[(fraction, seed, reference)]
            for fraction, seed, model, reference in
            deltas[["buy_fraction", "seed", "model", "reference"]].itertuples(index=False, name=None)
        ]
        write_csv(deltas, deltas_path)
    (study / "hit_rate_backfill.json").write_text(json.dumps({
        "metric": METRIC, "method": "mean(Recall@10 > 0) on saved rankings",
        "original_model_training": "unchanged",
    }, indent=2) + "\n")
    print(f"{study}: {len(rankings)} ranking tables, {len(test_runs)} test runs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, nargs="+")
    for folder in parser.parse_args().study:
        backfill(folder)
