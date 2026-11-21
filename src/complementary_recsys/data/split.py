from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import SplitBundle, pair_index


@dataclass(frozen=True)
class AmazonSplitter:
    seed: int = 42
    train_share: float = 0.8
    validation_share: float = 0.1

    def split(self, items: pd.DataFrame, pairs: pd.DataFrame) -> SplitBundle:
        if not (0 < self.train_share < 1 and 0 < self.validation_share
                and self.train_share + self.validation_share < 1):
            raise ValueError("Expected positive train, validation and test shares")

        pairs = pairs.sort_values(list(pairs.columns)).reset_index(drop=True)
        keys = pair_index(pairs)
        buy = pairs["relation"].eq("buy")
        buy_keys = keys[buy].unique().sort_values()
        order = np.random.default_rng(self.seed).permutation(len(buy_keys))
        first = int(len(order) * self.train_share)
        second = int(len(order) * (self.train_share + self.validation_share))
        validation_keys, test_keys = buy_keys[order[first:second]], buy_keys[order[second:]]
        holdout = validation_keys.append(test_keys).sort_values()
        train = pairs.loc[~keys.isin(holdout)].copy()
        known_items = pd.unique(pd.concat([train["source_id"], train["target_id"]]))
        known = pairs["source_id"].isin(known_items) & pairs["target_id"].isin(known_items)
        return SplitBundle(
            items=items.loc[items["item_id"].isin(known_items)].sort_values("item_id"),
            train=train,
            validation=pairs.loc[buy & known & keys.isin(validation_keys)].copy(),
            test=pairs.loc[buy & known & keys.isin(test_keys)].copy(),
            holdout=holdout.to_frame(index=False),
        )

    def from_processed(self, directory: Path) -> SplitBundle:
        return self.split(
            pd.read_csv(directory / "items.csv.gz", dtype=str, keep_default_na=False),
            pd.read_csv(directory / "pairs.csv.gz", dtype=str, keep_default_na=False),
        )
