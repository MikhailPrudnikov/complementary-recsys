import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from ..data.bundle import TrainingData
from .base import Query
from .graph_paths import relation_matrix


class HOSVD:
    def __init__(self, rank: int, seed: int, relation_rank: int = 2):
        if rank < 1 or relation_rank not in (1, 2):
            raise ValueError("Expected positive item rank and relation rank of 1 or 2")
        self.rank = rank
        self.seed = seed
        self.relation_rank = relation_rank

    def fit(self, data: TrainingData) -> None:
        layers = [relation_matrix(data, relation) for relation in range(2)]
        self.sources = self._factors(layers)
        self.targets = self._factors([layer.T for layer in layers])
        relation_gram = np.array([
            [first.multiply(second).sum() for second in layers] for first in layers
        ])
        _, relation_vectors = np.linalg.eigh(relation_gram)
        self.relations = relation_vectors[:, -self.relation_rank:]
        projected_layers = np.stack([self.sources.T @ (layer @ self.targets) for layer in layers])
        self.core = np.einsum("rc,rab->cab", self.relations, projected_layers)
        self.buy = np.einsum("c,cab->ab", self.relations[0], self.core)

    def _factors(self, layers: list[sparse.spmatrix]) -> np.ndarray:
        unfolded = sparse.hstack(layers, format="csr")
        if self.rank >= unfolded.shape[0]:
            return np.eye(unfolded.shape[0])
        vectors, values, _ = svds(unfolded, k=self.rank, rng=np.random.default_rng(self.seed),
                                 return_singular_vectors="u")
        return vectors[:, np.argsort(values)[::-1]]

    @property
    def parameter_count(self) -> int:
        return self.sources.size + self.targets.size + self.relations.size + self.core.size

    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        return self.targets[candidates] @ (self.sources[query.source] @ self.buy)
