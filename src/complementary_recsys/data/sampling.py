import numpy as np
import pandas as pd

from .bundle import pair_index


def subsample_buys(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("buy_fraction must be in (0, 1]")
    if fraction == 1:
        return train
    keys = pair_index(train)
    buy = train["relation"].eq("buy")
    buy_keys = keys[buy].unique().sort_values()
    order = np.random.default_rng(seed).permutation(len(buy_keys))
    selected = buy_keys[order[:int(len(order) * fraction)]]
    return train.loc[~buy | keys.isin(selected)]


class NegativeSampler:
    """Sample outside known co-occurrences, holdout pairs and self-pairs."""

    def __init__(self, n_items: int, positives: np.ndarray, holdout: np.ndarray):
        self.n_items = n_items
        self.blocked = np.unique(np.concatenate([
            positives[:, 0] * n_items + positives[:, 2],
            positives[:, 2] * n_items + positives[:, 0],
            holdout[:, 0] * n_items + holdout[:, 1],
            holdout[:, 1] * n_items + holdout[:, 0],
            np.arange(n_items) * (n_items + 1),
        ]))
        counts = np.bincount(self.blocked // n_items, minlength=n_items)
        self.available = counts < n_items

    def sample(self, sources: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.available[sources].all():
            raise ValueError("A positive source has no eligible negative targets")
        targets = rng.integers(self.n_items, size=len(sources))
        pending = np.arange(len(sources))
        # Use an explicit complement for the rare, nearly saturated rows.
        for _ in range(20):
            keys = sources[pending] * self.n_items + targets[pending]
            locations = self.blocked.searchsorted(keys).clip(max=len(self.blocked) - 1)
            pending = pending[self.blocked[locations] == keys]
            if not len(pending):
                return targets
            targets[pending] = rng.integers(self.n_items, size=len(pending))
        for source in np.unique(sources[pending]):
            indices = pending[sources[pending] == source]
            low, high = self.blocked.searchsorted([source * self.n_items, (source + 1) * self.n_items])
            allowed = np.setdiff1d(np.arange(self.n_items), self.blocked[low:high] % self.n_items)
            targets[indices] = rng.choice(allowed, size=len(indices))
        return targets
