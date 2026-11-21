import numpy as np
import torch

from ..data.bundle import TrainingData
from .base import Query
from .gradient import GradientRanker, TrainingConfig, normal
from .tucker import Tucker


class MIGTF(GradientRanker):
    """Frozen Tucker scores plus the Lorentz correction from equation (5)."""

    def __init__(self, config: TrainingConfig, use_views: bool = True, hyperbolic_rank: int = 4):
        super().__init__(config, use_views)
        if hyperbolic_rank < 1:
            raise ValueError("hyperbolic_rank must be positive")
        self.hyperbolic_rank = hyperbolic_rank
        self.euclidean = Tucker(config, use_views)

    def fit(self, data: TrainingData) -> None:
        self.euclidean.fit(data)
        self.euclidean.requires_grad_(False)
        rng = torch.Generator().manual_seed(self.config.seed + 1)
        self.initialize(data, rng)
        self.optimize(data)

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        self.items = normal((data.n_items, self.hyperbolic_rank), rng)
        self.relations = normal((self.n_relations, self.hyperbolic_rank), rng)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        source, relation, target, _ = edges.T
        return self.euclidean(edges) + self.correction(source, relation, target)

    def correction(self, source, relation, target):
        u, t, v = (self.items[source], self.relations[relation], self.items[target])
        return self.lorentz_score(u, t, v)

    @staticmethod
    def lorentz_score(u, t, v):
        # Lorentz coordinates at curvature -1, with spatial vectors stored as parameters.
        u0, t0, v0 = ((1 + x.square().sum(dim=-1)).sqrt() for x in (u, t, v))
        uv = (u * v).sum(dim=-1) - u0 * v0
        ut = (u * t).sum(dim=-1) - u0 * t0
        tv = (t * v).sum(dim=-1) - t0 * v0
        return (2 + t0 - u0 - v0 - uv + ut + tv) / (u0 * v0 + u0 * t0 + t0 * v0)

    @torch.inference_mode()
    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        ids = torch.as_tensor(candidates, device=self.items.device)
        return (self.euclidean.buy_scores(query.source, ids) + self.correction(query.source, 0, ids)).cpu().numpy()
