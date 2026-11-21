import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import torch

from .data.bundle import RELATIONS, SplitBundle, TrainingData, sha256 as file_sha256
from .data.sampling import subsample_buys
from .data.split import AmazonSplitter
from .evaluation.metrics import TOP10_METRIC_ORDER
from .evaluation.ranking import RankingEvaluator
from .models.base import Query
from .models.complex import ComplEx
from .models.cp import CP
from .models.distmult import DistMult
from .models.gradient import TrainingConfig
from .models.graph_paths import GraphPaths
from .models.hosvd import HOSVD
from .models.mig_tf import MIGTF
from .models.most_popular import MostPopular
from .models.tucker import Tucker
from .models.tucker4d import Tucker4D
from .models.two_tower import TwoTower


GRADIENT_MODELS = {
    "tucker": Tucker, "distmult": DistMult, "cp": CP, "complex": ComplEx,
    "mig_tf": MIGTF, "tucker4d": Tucker4D, "two_tower": TwoTower,
}


def source_fingerprint() -> str:
    package = Path(__file__).parent
    sources = {str(path.relative_to(package)): file_sha256(path)
               for path in sorted(package.rglob("*.py"))}
    return sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def build_model(spec: dict, training: TrainingConfig):
    options = spec.copy()
    options.pop("name")
    kind = options.pop("model")
    training = replace(training, **options.pop("training", {}))
    if kind == "most_popular":
        return MostPopular(**options)
    if kind == "graph_paths":
        return GraphPaths(**options)
    if kind == "hosvd":
        return HOSVD(training.rank, training.seed, **options)
    return GRADIENT_MODELS[kind](training, **options)


class ExperimentRunner:
    def __init__(self, bundle: SplitBundle, config: dict, partition: str):
        self.bundle = bundle
        self.config = config
        self.partition = partition
        self.training = TrainingConfig(**config.get("training", {}))
        self.personalized = config.get("personalized", False)

    def catalog(self) -> pd.DataFrame:
        items = self.bundle.items
        train = self.bundle.train
        known = set(train["source_id"]) | set(train["target_id"])
        items = items.loc[items["item_id"].isin(known) & items["category"].ne("")]
        limit = self.config.get("max_items")
        if limit is not None:
            if limit < 2:
                raise ValueError("max_items must be at least 2")
            buys = train.loc[train["relation"].eq("buy")]
            degree = pd.concat([buys["source_id"], buys["target_id"]]).value_counts()
            items = items.assign(degree=items["item_id"].map(degree).fillna(0))
            items = items.sort_values(["degree", "item_id"], ascending=[False, True]).head(limit)
            items = items.drop(columns="degree")
        return items.sort_values("item_id")

    def build_inputs(self):
        items = self.catalog()
        item_ids = items["item_id"].tolist()
        item_index = {value: key for key, value in enumerate(item_ids)}
        category_index = dict(zip(items["item_id"], items["category"]))
        train = self.bundle.train
        train = train.loc[train["source_id"].isin(item_index) & train["target_id"].isin(item_index)]
        if self.personalized:
            if "user_id" not in train or train["user_id"].eq("").any():
                raise ValueError("Personalized evaluation needs training user IDs")
            users = sorted(train["user_id"].unique())
        else:
            users = []
        user_index = {value: key for key, value in enumerate(users)}

        seen = defaultdict(set)
        for row in train.loc[train["relation"].eq("buy")].itertuples(index=False):
            user = user_index[row.user_id] if self.personalized else -1
            seen[item_index[row.source_id], user].add(item_index[row.target_id])

        evaluation = getattr(self.bundle, self.partition)
        targets = defaultdict(set)
        eligible_queries = set()
        for row in evaluation.itertuples(index=False):
            user_id = row.user_id if self.personalized else ""
            query_key = (row.source_id, row.target_category, user_id)
            eligible_queries.add(query_key)
            if row.source_id not in item_index or row.target_id not in item_index:
                continue
            if self.personalized and user_id not in user_index:
                continue
            if row.target_category != category_index[row.target_id]:
                continue
            query = Query(item_index[row.source_id], row.target_category, user_index.get(user_id, -1))
            targets[query].add(item_index[row.target_id])

        evaluator = RankingEvaluator(items["category"].to_numpy(), seen)
        cases = evaluator.build_cases(targets)
        if not cases:
            raise ValueError(f"No eligible {self.partition} ranking queries for this catalog")
        fraction = self.config.get("buy_fraction", 1)
        sampled = subsample_buys(train, fraction, self.config.get("subsample_seed", 42))
        user_ids = sampled["user_id"].map(user_index) if self.personalized else np.full(len(sampled), -1)
        edges = np.column_stack([
            sampled["source_id"].map(item_index), sampled["relation"].map(RELATIONS),
            sampled["target_id"].map(item_index), user_ids,
        ]).astype(np.int64)
        edges = np.unique(edges, axis=0)
        holdout = self.bundle.holdout
        if holdout is None:
            holdout_ids = np.empty((0, 2), dtype=np.int64)
        else:
            holdout_ids = holdout[["source_id", "target_id"]].apply(lambda column: column.map(item_index))
            holdout_ids = holdout_ids.dropna().to_numpy(dtype=np.int64)

        titles = items["title"] if "title" in items else [""] * len(items)
        data = TrainingData(len(items), edges, holdout_ids, len(users),
                            tuple(items["category"]), tuple(titles))
        return data, evaluator, cases, item_ids, users, len(eligible_queries)

    def run(self, output: Path) -> pd.DataFrame:
        data, evaluator, cases, item_ids, users, total_queries = self.build_inputs()
        specs = self.config["models"]
        names = [spec["name"] for spec in specs]
        if not names or len(set(names)) != len(names):
            raise ValueError("Expected at least one model and unique model names")
        if any(spec["model"] == "tucker4d" for spec in specs) and not self.personalized:
            raise ValueError("Use personalized=true and a temporal bundle for Tucker4D")
        output.mkdir(parents=True, exist_ok=False)
        pd.DataFrame({"item_id": item_ids}).to_csv(output / "catalog.csv", index=False)
        manifest = {
            "split_fingerprint": self.bundle.fingerprint, "partition": self.partition,
            "config": self.config, "training": asdict(self.training),
            "catalog_size": data.n_items, "users": data.n_users,
            "train_buy": int((data.edges[:, 1] == 0).sum()),
            "train_view": int((data.edges[:, 1] == 1).sum()),
            "eligible_queries": len(cases), "total_queries": total_queries,
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__,
            "device": (torch.cuda.get_device_name(self.training.device)
                       if self.training.device.startswith("cuda") else self.training.device),
            "code_revision": self.config.get("code_revision"),
            "source_fingerprint": source_fingerprint(),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        results, predictions = [], []
        for spec in specs:
            model = build_model(spec, self.training)
            start = perf_counter()
            model.fit(data)
            if self.training.device.startswith("cuda"):
                torch.cuda.synchronize(self.training.device)
            fit_time = perf_counter() - start
            start = perf_counter()
            metrics, rankings = evaluator.evaluate(model, cases)
            evaluation_time = perf_counter() - start
            row = {
                "model": spec["name"], **metrics, "coverage": len(cases) / total_queries,
                "parameters": model.parameter_count, "fit_seconds": fit_time,
                "evaluation_seconds": evaluation_time,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            for case, ranked, recall, ndcg, hit in rankings:
                query = case.query
                predictions.append({
                    "model": spec["name"], "source_id": item_ids[query.source],
                    "target_category": query.category,
                    "user_id": users[query.user] if query.user >= 0 else "",
                    "top10": json.dumps([item_ids[item] for item in ranked]),
                    "relevant": json.dumps([item_ids[item] for item in case.relevant]),
                    "recall@10": recall, "ndcg@10": ndcg, "hit_rate@10": hit,
                })
        results = pd.DataFrame(results).sort_values(
            TOP10_METRIC_ORDER, ascending=False, ignore_index=True,
        )
        results.to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(output / "recommendations.csv.gz", index=False)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed splits and evaluate complementary rankers")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split-amazon")
    split.add_argument("--processed", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("run")
    run.add_argument("--split", type=Path, required=True)
    run.add_argument("--items", type=Path, help="Original items.csv.gz for legacy Amazon splits")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--partition", choices=["validation", "test"], default="validation")
    run.add_argument("--device", help="Override the device in the experiment config")
    run.add_argument("--revision", help="Git commit or source snapshot identifier recorded in the manifest")
    args = parser.parse_args()

    if args.command == "split-amazon":
        AmazonSplitter(seed=args.seed).from_processed(args.processed).save(args.output, args.seed)
        return
    config = json.loads(args.config.read_text())
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    if args.revision:
        config["code_revision"] = args.revision
    bundle = SplitBundle.load(args.split, args.items)
    ExperimentRunner(bundle, config, args.partition).run(args.output)


if __name__ == "__main__":
    main()
