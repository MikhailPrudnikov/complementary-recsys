from dataclasses import replace
from hashlib import blake2b
import re

import numpy as np
import torch

from ..data.bundle import TrainingData
from .base import Query
from .gradient import GradientRanker, TrainingConfig, normal


class TwoTower(GradientRanker):
    """Matched feature encoder for buy-only, mixed and relation-specific heads."""

    def __init__(self, config: TrainingConfig, mode: str, title_buckets: int = 2048):
        if mode not in ("buy_only", "mixed", "shared"):
            raise ValueError(f"Unexpected two-tower mode: {mode}")
        if title_buckets < 1:
            raise ValueError("title_buckets must be positive")
        super().__init__(config, use_views=mode == "shared")
        self.exclude_holdout_from_negatives = mode != "mixed"
        self.mode = mode
        self.title_buckets = title_buckets

    def fit(self, data: TrainingData) -> None:
        if self.mode == "shared":
            buys = data.edges[data.edges[:, 1] == 0]
            views = data.edges[data.edges[:, 1] == 1]
            rng = np.random.default_rng(self.config.seed)
            views = views[rng.permutation(len(views))[:self.config.view_examples_per_epoch]]
            combined = np.concatenate([buys, views])
            data = replace(data, edges=np.unique(combined, axis=0))
        super().fit(data)

    def loss_steps(self, data: TrainingData, order_rngs, negative_rngs):
        if self.mode != "mixed":
            yield from super().loss_steps(data, order_rngs, negative_rngs)
            return

        config = self.config
        buys = data.edges[data.edges[:, 1] == 0]
        views = data.edges[data.edges[:, 1] == 1]
        if not len(buys):
            raise ValueError("Mixed requires positive buy edges")
        sampler = self.negative_sampler(data, data.edges)
        for _ in range(config.epochs):
            order = order_rngs[1].permutation(len(views))[:config.view_examples_per_epoch]
            combined = np.concatenate([buys, views[order]])
            combined[:, 1] = 0
            positives = np.unique(combined, axis=0)
            batch_order = order_rngs[0].permutation(len(positives))
            for start in range(0, len(positives), config.batch_size):
                batch = positives[batch_order[start:start + config.batch_size]]
                yield self.batch_loss(batch, sampler, negative_rngs[0])

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        if data.categories is None or data.titles is None:
            raise ValueError("TwoTower requires aligned catalog categories and titles")
        rank = self.config.rank
        categories = {name: index for index, name in enumerate(sorted(set(data.categories)))}
        self.category_ids = categories
        title_ids = np.full((data.n_items, 12), self.title_buckets, dtype=np.int64)
        for item, title in enumerate(data.titles):
            tokens = re.findall(r"[^\W_]+", title.casefold())[:title_ids.shape[1]]
            for column, token in enumerate(tokens):
                hashed = blake2b(token.encode(), digest_size=8).digest()
                title_ids[item, column] = int.from_bytes(hashed, "little") % self.title_buckets
        self.register_buffer("item_categories", torch.tensor([categories[name] for name in data.categories]))
        self.register_buffer("title_ids", torch.from_numpy(title_ids))
        self.items = normal((data.n_items, rank), rng)
        self.categories = normal((len(categories), rank), rng)
        self.words = normal((self.title_buckets + 1, rank), rng)
        self.category_projection = normal((rank, rank), rng)
        query_heads = [normal((rank, rank), rng).detach()]
        candidate_heads = [normal((rank, rank), rng).detach()]
        if self.n_relations == 2:
            query_heads.append(normal((rank, rank), rng).detach())
            candidate_heads.append(normal((rank, rank), rng).detach())
        self.query_heads = torch.nn.Parameter(torch.stack(query_heads))
        self.candidate_heads = torch.nn.Parameter(torch.stack(candidate_heads))

    def encode(self, items: torch.Tensor) -> torch.Tensor:
        tokens = self.title_ids[items]
        mask = tokens.ne(self.title_buckets).unsqueeze(-1)
        words = (self.words[tokens] * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1)
        return self.items[items] + self.categories[self.item_categories[items]] + words

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        return self.score_pairs(edges, self.item_categories[edges[:, 2]])

    @torch.inference_mode()
    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        if self.mode != "mixed":
            return super().score(query, candidates)
        category = self.category_ids.get(query.category)
        if category is None:
            raise ValueError(f"Unknown query category: {query.category}")
        edges = self.scoring_edges(query, candidates)
        target_categories = edges[:, 0].new_full((len(candidates),), category)
        return self.score_pairs(edges, target_categories).cpu().numpy()

    def training_logits(self, edges: torch.Tensor, n_positives: int) -> torch.Tensor:
        positive_categories = self.item_categories[edges[:n_positives, 2]]
        negative_categories = positive_categories.repeat_interleave(self.config.negatives)
        return self.score_pairs(edges, torch.cat([positive_categories, negative_categories]))

    def score_pairs(self, edges: torch.Tensor, query_categories: torch.Tensor) -> torch.Tensor:
        source, relation, target, _ = edges.T
        category = self.categories[query_categories]
        query = self.encode(source) + category @ self.category_projection
        candidate = self.encode(target)
        query = torch.einsum("bi,bij->bj", query, self.query_heads[relation])
        candidate = torch.einsum("bi,bij->bj", candidate, self.candidate_heads[relation])
        return (query * candidate).sum(dim=1)
