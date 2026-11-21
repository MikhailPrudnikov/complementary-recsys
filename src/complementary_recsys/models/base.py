from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..data.bundle import TrainingData


@dataclass(frozen=True)
class Query:
    source: int
    category: str
    user: int = -1


class Ranker(Protocol):
    @property
    def parameter_count(self) -> int: ...

    def fit(self, data: TrainingData) -> None: ...

    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray: ...
