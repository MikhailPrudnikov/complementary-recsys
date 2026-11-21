import torch

from ..data.bundle import TrainingData
from .gradient import GradientRanker, normal


class DistMult(GradientRanker):
    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        rank = self.config.rank
        self.items = normal((data.n_items, rank), rng)
        self.relations = normal((self.n_relations, rank), rng)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        source, relation, target, _ = edges.T
        return (self.items[source] * self.relations[relation] * self.items[target]).sum(dim=1)
