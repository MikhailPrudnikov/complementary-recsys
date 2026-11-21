import numpy as np
import torch

from ..data.bundle import TrainingData
from .base import Query
from .gradient import GradientRanker, normal


class Tucker(GradientRanker):
    """Shared product factors, with full rank on the relation axis (RESCAL scores)."""

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        rank = self.config.rank
        self.items = normal((data.n_items, rank), rng)
        self.relations = normal((self.n_relations, rank, rank), rng)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        source, relation, target, _ = edges.T
        return torch.einsum("bi,bij,bj->b", self.items[source], self.relations[relation], self.items[target])

    def buy_scores(self, source: int, candidates: torch.Tensor) -> torch.Tensor:
        return self.items[candidates] @ (self.items[source] @ self.relations[0])

    @torch.inference_mode()
    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        return self.buy_scores(query.source, torch.as_tensor(candidates, device=self.items.device)).cpu().numpy()
