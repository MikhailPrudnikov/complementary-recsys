"""Select model ranks on validation and summarize a three-seed local test study."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from complementary_recsys.data.bundle import SplitBundle
from complementary_recsys.evaluation.metrics import TOP10_MEAN_ORDER, TOP10_METRIC_ORDER
from complementary_recsys.experiment import ExperimentRunner, GRADIENT_MODELS


SEEDS = (42, 43, 44)
AMAZON_FRACTIONS = (1.0, 0.5, 0.25)
QUALITY_MARGIN = 0.01
VIEW_WEIGHTS = (0.25, 0.5, 1.0)
SHARED_RANK_GROUPS = (
    ("Tucker buy-only", "Tucker buy+view", "HOSVD"),
    ("TwoTower Buy-only", "TwoTower Mixed", "TwoTower Shared"),
    ("DistMult buy-only", "DistMult buy+view"),
    ("CP buy-only", "CP buy+view"),
    ("ComplEx buy-only", "ComplEx buy+view"),
    ("Tucker 3D", "Tucker 4D"),
)


def with_settings(base: dict, fraction: float, seed: int, models: list[dict]) -> dict:
    return {
        **base, "buy_fraction": fraction,
        "training": {**base["training"], "seed": seed},
        "models": models,
    }


def select_models(bundle: SplitBundle, config: dict, output: Path):
    original = config["models"]
    models = {spec["name"]: spec for spec in original}
    candidates = []
    rank_grid = (8, 16) if config.get("personalized", False) else (16, 32)
    for spec in original:
        ranks = rank_grid if spec["model"] in {*GRADIENT_MODELS, "hosvd"} else (None,)
        for rank in ranks:
            candidate = deepcopy(spec)
            if rank is not None:
                candidate["name"] = f"{spec['name']} [d={rank}]"
                candidate["training"] = {**candidate.get("training", {}), "rank": rank}
            candidates.append((spec["name"], rank, candidate))

    validation = ExperimentRunner(bundle, with_settings(
        config, 1, SEEDS[0], [candidate[2] for candidate in candidates],
    ), "validation").run(output / "validation")
    by_name = validation.set_index("model")["ndcg@10"]
    family_scores = {}
    for group in SHARED_RANK_GROUPS:
        group = [name for name in group if name in models]
        if len(group) > 1:
            for rank in rank_grid:
                score = sum(by_name.at[f"{name} [d={rank}]"] for name in group) / len(group)
                family_scores.update({(name, rank): score for name in group})
    selections = {}
    for name, rank, candidate in candidates:
        key = (family_scores.get((name, rank), by_name.at[candidate["name"]]), -(rank or 0))
        if name not in selections or key > selections[name][0]:
            selected = {**candidate, "name": name}
            selections[name] = (key, selected)
    selected_models = [selections[name][1] for name in models]
    weighted_candidates = []
    for spec in selected_models:
        name = spec["name"]
        if name in {"Tucker buy+view", "TwoTower Shared", "Graph mixed paths"}:
            for weight in VIEW_WEIGHTS:
                candidate = {**spec, "name": f"{name} [view={weight}]"}
                if spec["model"] == "graph_paths":
                    candidate["view_weight"] = weight
                else:
                    candidate["training"] = {**candidate["training"], "view_weight": weight}
                weighted_candidates.append((name, weight, candidate))
    if weighted_candidates:
        valid_weights = ExperimentRunner(bundle, with_settings(
            config, 1, SEEDS[0], [candidate[2] for candidate in weighted_candidates],
        ), "validation").run(output / "validation-view-weight")
        weight_scores = valid_weights.set_index("model")["ndcg@10"]
        weight_selections = {}
        for name, weight, candidate in weighted_candidates:
            key = (weight_scores.at[candidate["name"]], -weight)
            if name not in weight_selections or key > weight_selections[name][0]:
                weight_selections[name] = (key, {**candidate, "name": name})
        selected_models = [weight_selections[spec["name"]][1]
                           if spec["name"] in weight_selections else spec for spec in selected_models]
    return selected_models, rank_grid, VIEW_WEIGHTS if weighted_candidates else ()


def run_study(split: Path, config_path: Path, output: Path, fixed_settings: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    bundle = SplitBundle.load(split)
    config = json.loads(config_path.read_text())
    models = {spec["name"] for spec in config["models"]}
    if fixed_settings:
        selected_models, rank_grid, view_weight_grid = config["models"], (), ()
    else:
        selected_models, rank_grid, view_weight_grid = select_models(bundle, config, output)
    fractions = (1.0,) if config.get("personalized", False) else AMAZON_FRACTIONS

    # Write the exact selections and decision thresholds before evaluating test.
    protocol = {
        "split_fingerprint": bundle.fingerprint,
        "selection_metric": "fixed config" if fixed_settings else "validation NDCG@10",
        "validation_seed": None if fixed_settings else SEEDS[0], "test_seeds": SEEDS,
        "rank_grid": rank_grid, "shared_rank_groups": SHARED_RANK_GROUPS,
        "view_weight_grid": view_weight_grid,
        "buy_fractions": fractions, "quality_margin_absolute": QUALITY_MARGIN,
        "selected_models": selected_models,
    }
    (output / "selected_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    tables = []
    for fraction in fractions:
        for seed in SEEDS:
            results = ExperimentRunner(
                bundle, with_settings(config, fraction, seed, selected_models), "test",
            ).run(output / f"test-{int(100 * fraction)}-seed-{seed}")
            tables.append(results.assign(buy_fraction=fraction, seed=seed))
    runs = pd.concat(tables, ignore_index=True)
    runs.sort_values(TOP10_METRIC_ORDER, ascending=False).to_csv(output / "test_runs.csv", index=False)
    summary = runs.groupby(["buy_fraction", "model"], sort=False).agg(
        recall_mean=("recall@10", "mean"), recall_std=("recall@10", "std"),
        ndcg_mean=("ndcg@10", "mean"), ndcg_std=("ndcg@10", "std"),
        hit_rate_mean=("hit_rate@10", "mean"), hit_rate_std=("hit_rate@10", "std"),
        fit_seconds=("fit_seconds", "median"), evaluation_seconds=("evaluation_seconds", "median"),
        parameters=("parameters", "first"), queries=("queries", "first"), coverage=("coverage", "first"),
    )
    summary.sort_values(TOP10_MEAN_ORDER, ascending=False).to_csv(output / "test_summary.csv")
    pairs = [
        ("Tucker buy+view", "Tucker buy-only"),
        ("TwoTower Shared", "TwoTower Mixed"),
        ("TwoTower Shared", "TwoTower Buy-only"),
        ("Tucker buy+view", "TwoTower Shared"),
        ("Tucker buy+view", "HOSVD"),
        ("Tucker buy+view", "TwoTower Buy-only"),
        ("Tucker buy+view", "MostPopular"),
        ("Graph mixed paths", "Graph buy-buy"),
        ("DistMult buy+view", "DistMult buy-only"),
        ("Tucker 4D", "Tucker 3D"),
    ]
    wide = runs.pivot(index=["buy_fraction", "seed"], columns="model",
                     values=["ndcg@10", "recall@10", "hit_rate@10"])
    deltas = []
    for first, second in pairs:
        if first not in models or second not in models:
            continue
        difference = wide.xs(first, level="model", axis=1) - wide.xs(second, level="model", axis=1)
        deltas.append(difference.reset_index().assign(model=first, reference=second))
    if deltas:
        pd.concat(deltas, ignore_index=True).to_csv(output / "paired_deltas.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-settings", action="store_true",
                        help="Compare with existing settings without selecting new hyperparameters")
    args = parser.parse_args()
    run_study(args.split, args.config, args.output, args.fixed_settings)
