import torch

from ..data.bundle import TrainingData
from .gradient import GradientRanker, normal


class CP(GradientRanker):
    """Canonical polyadic factorization with separate source and candidate factors."""

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        rank = self.config.rank
        self.sources = normal((data.n_items, rank), rng)
        self.targets = normal((data.n_items, rank), rng)
        self.relations = normal((self.n_relations, rank), rng)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        source, relation, target, _ = edges.T
        return (self.sources[source] * self.relations[relation] * self.targets[target]).sum(dim=1)
