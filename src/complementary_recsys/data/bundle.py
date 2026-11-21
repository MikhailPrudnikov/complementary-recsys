from dataclasses import dataclass
from hashlib import file_digest, sha256 as digest
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_COLUMNS = ["source_id", "target_id"]
RELATIONS = {"buy": 0, "view": 1}


def pair_index(frame: pd.DataFrame) -> pd.MultiIndex:
    source, target = frame["source_id"], frame["target_id"]
    forward = source <= target
    return pd.MultiIndex.from_arrays(
        [source.where(forward, target), target.where(forward, source)],
        names=PAIR_COLUMNS,
    )


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class SplitBundle:
    items: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame | None
    fingerprint: str | None = None

    @classmethod
    def load(cls, directory: Path, items_path: Path | None = None) -> "SplitBundle":
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        paths = {name: directory / f"{name}.csv.gz" for name in ("train", "validation", "test")}
        holdout = directory / "holdout_pairs.csv.gz"
        if holdout.is_file():
            paths["holdout"] = holdout
        if items_path is None and (directory / "items.csv.gz").is_file():
            items_path = directory / "items.csv.gz"
        if items_path is not None:
            paths["items"] = items_path
        elif (directory / "train_items.csv.gz").is_file():
            paths["train_items"] = directory / "train_items.csv.gz"
        hashes = {name: sha256(path) for name, path in paths.items()}
        if manifest is not None and manifest["files"] != hashes:
            raise ValueError("Split files do not match manifest.json")
        frames = {name: pd.read_csv(path, dtype=str, keep_default_na=False)
                  for name, path in paths.items()}
        if "items" not in frames:
            # Event labels can change. Use the most frequent training category per ID.
            train = frames["train"]
            categories = pd.concat([
                train[["source_id", "source_category"]].set_axis(["item_id", "category"], axis=1),
                train[["target_id", "target_category"]].set_axis(["item_id", "category"], axis=1),
            ])
            categories = categories.groupby(["item_id", "category"]).size().rename("count").reset_index()
            frames["items"] = (
                categories.sort_values(["count", "category"], ascending=[False, True])
                .drop_duplicates("item_id")[["item_id", "category"]]
            )
            if "train_items" in frames:
                frames["items"] = frames["items"].loc[
                    frames["items"]["item_id"].isin(frames["train_items"]["item_id"])
                ]
        frames.pop("train_items", None)
        frames.setdefault("holdout", None)
        fingerprint = digest(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = cls(**frames, fingerprint=fingerprint)
        bundle.validate()
        return bundle

    def save(self, directory: Path, split_seed: int) -> None:
        self.validate()
        directory.mkdir(parents=True, exist_ok=False)
        frames = {
            "items": self.items, "train": self.train,
            "validation": self.validation, "test": self.test,
        }
        if self.holdout is not None:
            frames["holdout_pairs"] = self.holdout
        hashes = {}
        for name, frame in frames.items():
            path = directory / f"{name}.csv.gz"
            frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            hashes["holdout" if name == "holdout_pairs" else name] = sha256(path)
        (directory / "manifest.json").write_text(
            json.dumps({"split_seed": split_seed, "files": hashes}, indent=2, sort_keys=True) + "\n"
        )

    def validate(self) -> None:
        if self.items["item_id"].duplicated().any():
            raise ValueError("Catalog contains duplicate item IDs")
        for name in ("train", "validation", "test"):
            frame = getattr(self, name)
            relations = RELATIONS if name == "train" else {"buy"}
            if frame.empty or not frame["relation"].isin(relations).all():
                raise ValueError(f"Invalid relations or an empty {name} split")
        if self.holdout is not None:
            holdout = pair_index(self.holdout)
            validation, test = pair_index(self.validation), pair_index(self.test)
            if (pair_index(self.train).isin(holdout).any() or validation.isin(test).any()
                    or not validation.isin(holdout).all() or not test.isin(holdout).all()):
                raise ValueError("Amazon pair split overlaps or has an incomplete holdout manifest")


@dataclass(frozen=True)
class TrainingData:
    n_items: int
    edges: np.ndarray  # source, relation, target, user (-1 for static graphs)
    holdout: np.ndarray
    n_users: int
    categories: tuple[str, ...] | None = None
    titles: tuple[str, ...] | None = None
