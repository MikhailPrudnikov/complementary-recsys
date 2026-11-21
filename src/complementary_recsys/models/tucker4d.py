import numpy as np
import torch

from ..data.bundle import TrainingData
from .base import Query
from .gradient import GradientRanker, TrainingConfig, normal


class Tucker4D(GradientRanker):
    """Tucker with learned user factors and a shared product matrix."""

    def __init__(self, config: TrainingConfig, use_views: bool = True, user_rank: int = 4):
        super().__init__(config, use_views)
        if user_rank < 1:
            raise ValueError("user_rank must be positive")
        self.user_rank = user_rank

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        if data.n_users == 0 or (data.edges[:, 3] < 0).any():
            raise ValueError("Tucker4D requires a temporal split with user IDs")
        rank = self.config.rank
        self.items = normal((data.n_items, rank), rng)
        self.users = normal((data.n_users, self.user_rank), rng)
        self.core = normal((self.n_relations, rank, rank, self.user_rank), rng)

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        source, relation, target, user = edges.T
        return torch.einsum(
            "bi,bijk,bj,bk->b", self.items[source], self.core[relation],
            self.items[target], self.users[user],
        )

    @torch.inference_mode()
    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        if query.user < 0:
            raise ValueError("Tucker4D requires a user observed in the training graph")
        weights = torch.einsum("i,ijk,k->j", self.items[query.source], self.core[0], self.users[query.user])
        ids = torch.as_tensor(candidates, device=self.items.device)
        return (self.items[ids] @ weights).cpu().numpy()
