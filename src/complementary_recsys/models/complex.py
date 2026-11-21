import torch

from ..data.bundle import TrainingData
from .gradient import GradientRanker, normal


class ComplEx(GradientRanker):
    """Real part of the trilinear complex product with a conjugated target."""

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        rank = self.config.rank
        self.items = normal((2, data.n_items, rank), rng)
        self.relations = normal((2, self.n_relations, rank), rng)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        source, relation, target, _ = edges.T
        sr, si = self.items[:, source]
        tr, ti = self.items[:, target]
        rr, ri = self.relations[:, relation]
        return ((sr * rr - si * ri) * tr + (sr * ri + si * rr) * ti).sum(dim=1)
