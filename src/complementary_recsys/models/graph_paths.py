import numpy as np
from scipy import sparse

from ..data.bundle import TrainingData
from .base import Query


def relation_matrix(data: TrainingData, relation: int) -> sparse.csr_matrix:
    source, _, target, _ = data.edges[data.edges[:, 1] == relation].T
    matrix = sparse.csr_matrix(
        (np.ones(len(source)), (source, target)), shape=(data.n_items, data.n_items)
    )
    matrix.data[:] = 1
    return matrix


def normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1, degree, out=np.zeros_like(degree), where=degree != 0)
    return matrix.multiply(inverse[:, None]).tocsr()


class GraphPaths:
    parameter_count = 0

    def __init__(self, view_weight: float):
        if view_weight < 0:
            raise ValueError("view_weight must be nonnegative")
        self.view_weight = view_weight

    def fit(self, data: TrainingData) -> None:
        self.buy = normalize_rows(relation_matrix(data, 0))
        if self.view_weight:
            self.view = normalize_rows(relation_matrix(data, 1))

    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        buy_row = self.buy[query.source]
        score = buy_row @ self.buy
        if self.view_weight:
            score = score + self.view_weight * (buy_row @ self.view + self.view[query.source] @ self.buy)
        return score[:, candidates].toarray().ravel()
