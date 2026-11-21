import numpy as np


TOP10_METRIC_ORDER = ["hit_rate@10", "ndcg@10", "recall@10"]
TOP10_MEAN_ORDER = ["hit_rate_mean", "ndcg_mean", "recall_mean"]


def ranking_metrics(ranked: np.ndarray, relevant: np.ndarray, k: int = 10) -> tuple[float, float, int]:
    hits = np.isin(ranked[:k], relevant)
    discounts = 1 / np.log2(np.arange(2, 2 + k))
    recall = hits.sum() / len(relevant)
    dcg = hits @ discounts[:len(hits)]
    ideal = discounts[:min(k, len(relevant))].sum()
    return float(recall), float(dcg / ideal), int(hits.any())
