from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..models.base import Query, Ranker
from .metrics import ranking_metrics


@dataclass(frozen=True)
class RankingCase:
    query: Query
    candidates: np.ndarray
    relevant: np.ndarray


class RankingEvaluator:
    def __init__(self, categories: np.ndarray, seen: dict[tuple[int, int], set[int]], k: int = 10):
        self.categories = categories
        self.seen = seen
        self.k = k
        self.by_category = defaultdict(list)
        for item, category in enumerate(categories):
            self.by_category[category].append(item)
        self.by_category = {category: np.asarray(items, dtype=np.int64)
                            for category, items in self.by_category.items()}

    def build_cases(self, targets: dict[Query, set[int]]) -> list[RankingCase]:
        cases = []
        for query, relevant in sorted(targets.items(), key=lambda row: (
            row[0].source, row[0].category, row[0].user
        )):
            candidates = self.by_category.get(query.category, np.empty(0, dtype=np.int64))
            blocked = self.seen.get((query.source, query.user), set()) | {query.source}
            candidates = candidates[~np.isin(candidates, list(blocked))]
            eligible = np.intersect1d(candidates, list(relevant))
            if eligible.size:
                cases.append(RankingCase(query, candidates, eligible))
        return cases

    def evaluate(self, model: Ranker, cases: list[RankingCase]) -> tuple[dict, list]:
        if not cases:
            raise ValueError("No evaluation queries with eligible relevant candidates")
        rows = []
        for case in cases:
            scores = np.asarray(model.score(case.query, case.candidates))
            if scores.shape != case.candidates.shape or not np.isfinite(scores).all():
                raise ValueError(f"Invalid scores for query {case.query}")
            order = np.lexsort((case.candidates, -scores.astype(np.float64)))[:self.k]
            ranked = case.candidates[order]
            recall, ndcg, hit = ranking_metrics(ranked, case.relevant, self.k)
            rows.append((case, ranked, recall, ndcg, hit))
        return {
            "recall@10": float(np.mean([row[2] for row in rows])),
            "ndcg@10": float(np.mean([row[3] for row in rows])),
            "hit_rate@10": float(np.mean([row[4] for row in rows])),
            "queries": len(rows),
            "mean_candidates": float(np.mean([len(row[0].candidates) for row in rows])),
        }, rows
