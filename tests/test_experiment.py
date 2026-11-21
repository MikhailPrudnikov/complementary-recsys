import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from complementary_recsys.data.bundle import SplitBundle, TrainingData, pair_index
from complementary_recsys.data.sampling import NegativeSampler, subsample_buys
from complementary_recsys.data.split import AmazonSplitter
from complementary_recsys.evaluation.metrics import TOP10_METRIC_ORDER, ranking_metrics
from complementary_recsys.experiment import ExperimentRunner, build_model
from complementary_recsys.models.base import Query
from complementary_recsys.models.gradient import TrainingConfig
from complementary_recsys.models.two_tower import TwoTower


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pair_data():
    ids = [f"item-{index:02}" for index in range(30)]
    items = pd.DataFrame({"item_id": ids, "category": "accessories"})
    edges = []
    for relation, offsets in (("buy", (1, 2)), ("view", (1, 3, 4))):
        for index, source in enumerate(ids):
            for offset in offsets:
                target = ids[(index + offset) % len(ids)]
                edges.extend((source, target, relation, "accessories", "accessories")
                             for source, target in ((source, target), (target, source)))
    pairs = pd.DataFrame(edges, columns=["source_id", "target_id", "relation",
                                         "source_category", "target_category"])
    return items, pairs


def test_split_is_order_independent_and_negatives_exclude_holdout(pair_data, tmp_path):
    items, pairs = pair_data
    splitter = AmazonSplitter()
    first = splitter.split(items, pairs)
    second = splitter.split(items.iloc[::-1], pairs.iloc[::-1])
    for part in ("items", "train", "validation", "test", "holdout"):
        pd.testing.assert_frame_equal(getattr(first, part), getattr(second, part))

    first.save(tmp_path / "v1", split_seed=42)
    second.save(tmp_path / "v2", split_seed=42)
    bundle = SplitBundle.load(tmp_path / "v1")
    assert bundle.fingerprint == SplitBundle.load(tmp_path / "v2").fingerprint
    assert not pair_index(bundle.train).isin(pair_index(bundle.holdout)).any()

    def buys(frame):
        return set(pair_index(frame.loc[frame["relation"].eq("buy")]))

    assert buys(subsample_buys(bundle.train, 0.25, 42)) < buys(subsample_buys(bundle.train, 0.5, 42))

    sampler = NegativeSampler(6, np.array([[0, 0, 1, -1]]), np.array([[2, 0]]))
    negatives = sampler.sample(np.zeros(100, dtype=int), np.random.default_rng(42))
    assert set(negatives) == {3, 4, 5}


def test_metric_formula_and_ties(pair_data, tmp_path):
    recall, ndcg, hit = ranking_metrics(np.array([4, 2, 1]), np.array([2, 8]), k=2)
    assert recall == 0.5
    assert ndcg == pytest.approx(1 / (1 + np.log2(3)))
    assert hit == 1
    assert ranking_metrics(np.array([4, 1, 2]), np.array([2]), k=2) == (0.0, 0.0, 0)

    bundle = AmazonSplitter().split(*pair_data)
    config = {"models": [{"name": "pop", "model": "most_popular"}]}
    _, evaluator, cases, *_ = ExperimentRunner(bundle, config, "validation").build_inputs()

    class ConstantScores:
        def score(self, query, candidates):
            return np.zeros(len(candidates))

    _, rankings = evaluator.evaluate(ConstantScores(), cases)
    for case, ranked, *_ in rankings:
        np.testing.assert_array_equal(ranked, np.sort(case.candidates)[:10])


def test_all_amazon_runs_are_reproducible(pair_data, tmp_path):
    bundle = AmazonSplitter().split(*pair_data)
    config = json.loads((ROOT / "configs/amazon_pilot.json").read_text())
    config["training"].update(rank=4, epochs=1, batch_size=32)
    runner = ExperimentRunner(bundle, config, "validation")
    results = runner.run(tmp_path / "one")
    runner.run(tmp_path / "two")
    assert len(results) == 13
    assert results[TOP10_METRIC_ORDER].equals(
        results[TOP10_METRIC_ORDER].sort_values(TOP10_METRIC_ORDER, ascending=False, ignore_index=True)
    )
    assert results[["ndcg@10", "recall@10", "hit_rate@10"]].apply(
        lambda column: column.between(0, 1)
    ).all().all()
    assert pd.read_csv(tmp_path / "one/recommendations.csv.gz").equals(
        pd.read_csv(tmp_path / "two/recommendations.csv.gz")
    )


def test_personalized_models_share_queries(pair_data, tmp_path):
    amazon = AmazonSplitter().split(*pair_data)
    parts = {name: getattr(amazon, name).assign(user_id="test-user") for name in ("train", "validation", "test")}
    bundle = SplitBundle(amazon.items, **parts, holdout=None)
    config = json.loads((ROOT / "configs/temporal_pilot.json").read_text())
    config["training"].update(rank=4, epochs=1, batch_size=32)
    table = ExperimentRunner(bundle, config, "validation").run(tmp_path / "personalized")
    assert table["queries"].nunique() == 1
    parameters = table.set_index("model")["parameters"]
    assert parameters["Tucker 4D"] > parameters["Tucker 3D"]


def test_two_tower_modes_share_the_catalog_and_evaluator(pair_data, tmp_path):
    items, pairs = pair_data
    items = items.assign(title="guitar case")
    bundle = AmazonSplitter().split(items, pairs)
    config = json.loads((ROOT / "configs/filipp_main.json").read_text())
    config["models"] = [spec for spec in config["models"] if spec["model"] == "two_tower"]
    config["training"].update(rank=4, epochs=1, batch_size=32)

    table = ExperimentRunner(bundle, config, "validation").run(tmp_path / "two-tower")

    assert table["queries"].nunique() == 1
    parameters = table.set_index("model")["parameters"]
    assert parameters["TwoTower Buy-only"] == parameters["TwoTower Mixed"]
    assert parameters["TwoTower Shared"] - parameters["TwoTower Mixed"] == 2 * 4**2


def test_two_tower_keeps_the_query_category_for_negative_candidates(monkeypatch):
    model = TwoTower(TrainingConfig(negatives=2), "mixed")
    model.item_categories = torch.tensor([2, 0, 1])
    edges = torch.tensor([[0, 0, 1, -1], [0, 0, 2, -1], [0, 0, 0, -1]])
    monkeypatch.setattr(model, "score_pairs", lambda edges, categories: categories)

    assert model.training_logits(edges, n_positives=1).tolist() == [0, 0, 0]


def test_mixed_resamples_views_and_blocks_all_observed_training_pairs(monkeypatch):
    config = TrainingConfig(view_examples_per_epoch=1, epochs=3, batch_size=16)
    model = TwoTower(config, "mixed")
    edges = np.array([[0, 0, 1, -1], [0, 1, 2, -1], [0, 1, 3, -1],
                      [0, 1, 4, -1], [0, 1, 5, -1]])
    data = TrainingData(7, edges, np.array([[0, 6]]), 0)
    batches = []

    def batch_loss(positives, sampler, rng):
        batches.append(positives.copy())
        assert set(sampler.sample(np.zeros(12, dtype=int), rng)) == {6}
        return 0

    monkeypatch.setattr(model, "batch_loss", batch_loss)
    streams = tuple(np.random.default_rng(seed) for seed in (42, 43))
    list(model.loss_steps(data, streams, streams))

    assert len(batches) == config.epochs
    assert all(len(batch) == 2 and set(batch[:, 1]) == {0} and 1 in batch[:, 2] for batch in batches)
    assert len({tuple(sorted(batch[:, 2])) for batch in batches}) > 1


def test_mixed_scoring_uses_query_category_with_same_candidates(monkeypatch):
    model = TwoTower(TrainingConfig(), "mixed")
    model.category_ids = {"cases": 0, "cables": 1}
    monkeypatch.setattr(model, "score_pairs", lambda edges, categories: categories)
    candidates = np.array([1, 2])
    assert model.score(Query(0, "cases"), candidates).tolist() == [0, 0]
    assert model.score(Query(0, "cables"), candidates).tolist() == [1, 1]


def test_train_only_policy_for_owned_models_preserves_comparison_controls():
    edges = np.array([[0, 0, 1, -1]])
    data = TrainingData(4, edges, np.array([[0, 2]]), 0)
    config = TrainingConfig()
    owned = [
        {"name": name, "model": name}
        for name in ("tucker", "tucker4d", "distmult", "cp", "complex", "mig_tf")
    ]
    owned.append({"name": "Mixed", "model": "two_tower", "mode": "mixed"})
    controls = [{"name": mode, "model": "two_tower", "mode": mode} for mode in ("buy_only", "shared")]
    for spec, expected in [(spec, {2, 3}) for spec in owned] + [(spec, {3}) for spec in controls]:
        model = build_model(spec, config)
        sampler = model.negative_sampler(data, edges)
        assert set(sampler.sample(np.zeros(100, dtype=int), np.random.default_rng(42))) == expected
