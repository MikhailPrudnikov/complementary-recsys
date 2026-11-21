from dataclasses import dataclass
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from ..data.bundle import TrainingData
from ..data.sampling import NegativeSampler
from .base import Query


@dataclass(frozen=True)
class TrainingConfig:
    rank: int = 32
    epochs: int = 5
    batch_size: int = 512
    negatives: int = 3
    learning_rate: float = 0.01
    view_weight: float = 0.5
    view_examples_per_epoch: int = 4096
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        if min(self.rank, self.epochs, self.batch_size, self.negatives, self.view_examples_per_epoch) < 1:
            raise ValueError("Ranks, epochs, batch size, negative count and view budget must be positive")
        if self.learning_rate <= 0 or self.view_weight < 0:
            raise ValueError("Expected a positive learning rate and nonnegative view weight")


def normal(shape: tuple[int, ...], rng: torch.Generator, scale: float = 0.1) -> nn.Parameter:
    return nn.Parameter(torch.randn(shape, generator=rng) * scale)


class GradientRanker(nn.Module):
    def __init__(self, config: TrainingConfig, use_views: bool = True):
        super().__init__()
        self.config = config
        self.n_relations = 2 if use_views else 1
        self.exclude_holdout_from_negatives = False

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initialize(self, data: TrainingData, rng: torch.Generator) -> None:
        raise NotImplementedError

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, data: TrainingData) -> None:
        rng = torch.Generator().manual_seed(self.config.seed)
        self.initialize(data, rng)
        self.optimize(data)

    def optimize(self, data: TrainingData) -> None:
        config = self.config
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if config.device == "cpu":
            torch.set_num_threads(1)
        self.to(config.device)
        self.train()
        streams = [np.random.default_rng(seed) for seed in np.random.SeedSequence(config.seed).spawn(4)]
        order_rngs, negative_rngs = streams[:2], streams[2:]
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
        )
        for loss in self.loss_steps(data, order_rngs, negative_rngs):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.eval()

    def negative_sampler(self, data: TrainingData, positives: np.ndarray) -> NegativeSampler:
        holdout = data.holdout if self.exclude_holdout_from_negatives else data.holdout[:0]
        return NegativeSampler(data.n_items, positives, holdout)

    def loss_steps(self, data: TrainingData, order_rngs, negative_rngs):
        config = self.config
        groups = [data.edges[data.edges[:, 1] == relation] for relation in range(self.n_relations)]
        if not all(len(group) for group in groups):
            raise ValueError("The selected training relations must each contain positive edges")
        samplers = [self.negative_sampler(data, group) for group in groups]

        for _ in range(config.epochs):
            n_buys = len(groups[0])
            n_steps = (n_buys + config.batch_size - 1) // config.batch_size
            buy_order = order_rngs[0].permutation(n_buys)
            if self.n_relations == 2:
                view_order = order_rngs[1].permutation(len(groups[1]))[:config.view_examples_per_epoch]
                view_batches = np.array_split(view_order, n_steps)

            for step, start in enumerate(range(0, n_buys, config.batch_size)):
                buys = groups[0][buy_order[start:start + config.batch_size]]
                loss = self.batch_loss(buys, samplers[0], negative_rngs[0])
                if self.n_relations == 2 and len(view_batches[step]):
                    views = groups[1][view_batches[step]]
                    loss = loss + config.view_weight * self.batch_loss(views, samplers[1], negative_rngs[1])
                yield loss

    def batch_loss(self, positives: np.ndarray, sampler: NegativeSampler, rng: np.random.Generator):
        negatives = np.repeat(positives, self.config.negatives, axis=0)
        negatives[:, 2] = sampler.sample(negatives[:, 0], rng)
        edges = torch.as_tensor(np.concatenate([positives, negatives]), device=self.config.device)
        logits = self.training_logits(edges, len(positives))
        labels = torch.zeros_like(logits)
        labels[:len(positives)] = 1
        return binary_cross_entropy_with_logits(logits, labels)

    def training_logits(self, edges: torch.Tensor, n_positives: int) -> torch.Tensor:
        return self(edges)

    def scoring_edges(self, query: Query, candidates: np.ndarray) -> torch.Tensor:
        edges = np.column_stack([
            np.full(len(candidates), query.source), np.zeros(len(candidates), dtype=np.int64),
            candidates, np.full(len(candidates), query.user),
        ])
        return torch.as_tensor(edges, device=self.config.device)

    @torch.inference_mode()
    def score(self, query: Query, candidates: np.ndarray) -> np.ndarray:
        return self(self.scoring_edges(query, candidates)).cpu().numpy()
