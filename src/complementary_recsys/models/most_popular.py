import numpy as np

from ..data.bundle import TrainingData
from .base import Query


class MostPopular:
    parameter_count = 0

    def fit(self, data: TrainingData) -> None:
        targets = data.edges[data.edges[:, 1] == 0, 2]
        self.counts = np.bincount(targets, minlength=data.n_items)

    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        return self.counts[candidates]
