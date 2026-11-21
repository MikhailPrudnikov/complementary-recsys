"""Prepare an Amazon pilot, preserving the upstream grouped split before filtering."""

import argparse
import gzip
import hashlib
import html
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", html.unescape(str(text)).lower())


def pair_keys(frame):
    a, b = frame.source_id, frame.target_id
    return pd.MultiIndex.from_arrays([a.where(a <= b, b), b.where(a <= b, a)])


def raw_tables(path):
    rows = {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if r["asin"] in rows:
                continue  # Same first-record policy as the team's notebook.
            cats = r.get("category") or []
            rows[r["asin"]] = {
                "item_id": r["asin"],
                "title": r.get("title") or "",
                "category": cats[-1] if cats else None,
                "category_path": " > ".join(cats),
                "buy": r.get("also_buy") or [],
                "view": r.get("also_view") or [],
            }
    items = pd.DataFrame(rows.values())
    parts = []
    for relation in ["buy", "view"]:
        p = items[["item_id", relation]].explode(relation)
        p.columns = ["source_id", "target_id"]
        p = p.dropna().drop_duplicates()
        p = p[p.source_id.ne(p.target_id) & p.target_id.isin(items.item_id)]
        parts.append(p.assign(relation=relation))
    pairs = pd.concat(parts, ignore_index=True)
    categories = items.set_index("item_id").category
    pairs["source_category"] = pairs.source_id.map(categories)
    pairs["target_category"] = pairs.target_id.map(categories)
    return items.drop(columns=["buy", "view"]), pairs.dropna(
        subset=["source_category", "target_category"]
    )


def grouped_split(pairs, seed):
    buy = pairs[pairs.relation.eq("buy")]
    keys = pair_keys(buy).unique().sort_values()
    order = np.random.default_rng(seed).permutation(len(keys))
    a, b = int(len(keys) * 0.8), int(len(keys) * 0.9)
    valid, test = keys[order[a:b]], keys[order[b:]]
    holdout = valid.append(test)
    train = pairs[~pair_keys(pairs).isin(holdout)]
    known = set(train.source_id) | set(train.target_id)
    frames = {
        "train": train,
        "validation": buy[pair_keys(buy).isin(valid)],
        "test": buy[pair_keys(buy).isin(test)],
    }
    for name in ["validation", "test"]:
        p = frames[name]
        frames[name] = p[p.source_id.isin(known) & p.target_id.isin(known)]
    return frames, holdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        help="Optional existing Amazon split from the team notebook",
    )
    parser.add_argument("--max-items", type=int, default=20000)
    parser.add_argument("--min-category-items", type=int, default=20)
    parser.add_argument("--max-vocab", type=int, default=16000)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output is not empty; choose a new versioned directory")
    if not args.raw.exists():
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.raw.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(args.raw)
    items, pairs = raw_tables(args.raw)
    frames, holdout = grouped_split(pairs, args.seed)
    upstream_counts = {n: len(p) for n, p in frames.items()}
    if args.split_dir:
        frames = {
            n: pd.read_csv(args.split_dir / f"{n}.csv.gz", dtype="string")
            for n in ["train", "validation", "test"]
        }
        h = pd.read_csv(args.split_dir / "holdout_pairs.csv.gz", dtype="string")
        holdout = pair_keys(h).unique()
        upstream_counts = {n: len(p) for n, p in frames.items()}
        if pair_keys(frames["train"]).isin(holdout).any():
            raise ValueError("Holdout overlaps training data")
    print("Upstream split:", upstream_counts, flush=True)
    train = frames["train"]
    known = set(train.source_id) | set(train.target_id)
    items = items[items.item_id.isin(known)].copy()
    items["title"] = items.title.map(lambda s: html.unescape(str(s)).strip())
    items["category"] = items.category.map(
        lambda s: html.unescape(str(s)).strip() if pd.notna(s) else ""
    )
    items = items[items.title.ne("") & items.category.ne("")]
    # Catalog selection uses training relations only; test labels never select products.
    degree = pd.concat([train.source_id, train.target_id]).value_counts()
    items["degree"] = items.item_id.map(degree).fillna(0)
    items = items.sort_values(["degree", "item_id"], ascending=[False, True]).head(
        args.max_items
    )
    counts = items.category.value_counts()
    items = items[items.category.isin(counts[counts >= args.min_category_items].index)]
    items = items.sort_values("item_id").reset_index(drop=True)
    ids = {v: i for i, v in enumerate(items.item_id)}
    cats = {v: i for i, v in enumerate(sorted(items.category.unique()))}
    item_cats = np.array([cats[c] for c in items.category], dtype=np.int64)
    arrays = {"categories": item_cats}
    coverage = {}
    for name, frame in frames.items():
        p = frame[frame.source_id.isin(ids) & frame.target_id.isin(ids)].copy()
        p["source"] = p.source_id.map(ids)
        p["target"] = p.target_id.map(ids)
        # Conservative complementary proxy for the pilot; not a compatibility label.
        cross = (
            item_cats[p.source.to_numpy(dtype=int)]
            != item_cats[p.target.to_numpy(dtype=int)]
        )
        p = p[p.relation.eq("view") | cross]
        coverage[name] = {"before": len(frame), "retained": len(p)}
        for relation in ["buy", "view"] if name == "train" else ["buy"]:
            edge = (
                p[p.relation.eq(relation)][["source", "target"]]
                .drop_duplicates()
                .to_numpy(dtype=np.int64)
            )
            arrays[f"{name}_{relation}"] = edge.reshape(-1, 2)
    # All held-out unordered pairs, including filtered rows, remain forbidden negatives.
    blocked = [(ids[a], ids[b]) for a, b in holdout if a in ids and b in ids]
    arrays["holdout"] = np.array(blocked, dtype=np.int64).reshape(-1, 2)
    vocab_counts = Counter(t for title in items.title for t in tokenize(title))
    words = sorted(vocab_counts, key=lambda w: (-vocab_counts[w], w))[
        : args.max_vocab - 2
    ]
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tokens = np.zeros((len(items), args.max_tokens), dtype=np.int64)
    for i, title in enumerate(items.title):
        values = [vocab.get(w, 1) for w in tokenize(title)[: args.max_tokens]] or [1]
        tokens[i, : len(values)] = values
    arrays["tokens"] = tokens
    if any(
        len(arrays[k]) == 0
        for k in ["train_buy", "train_view", "validation_buy", "test_buy"]
    ):
        raise ValueError("Empty relation/split after filtering")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "benchmark.npz", **arrays)
    items.drop(columns="degree").to_csv(args.output / "items.csv.gz", index=False)
    (args.output / "vocabulary.json").write_text(json.dumps(vocab, ensure_ascii=False))
    manifest = {
        "dataset": "amazon_musical_instruments",
        "status": "pilot",
        "upstream_commit": "ae2ed6de8aa4832786fa08887c82db665573db69",
        "source_url": URL,
        "split_seed": args.seed,
        "upstream_counts": upstream_counts,
        "catalog_selection": "top train total degree, then category size >= minimum",
        "buy_scope": "different normalized leaf categories; heuristic only",
        "candidate_protocol": "all retained items in target category; mask full buy-train and self",
        "n_items": len(items),
        "n_categories": len(cats),
        "vocab_size": len(vocab),
        "category_names": list(cats),
        "coverage": coverage,
        "counts": {
            k: len(v) for k, v in arrays.items() if k not in ["tokens", "categories"]
        },
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "benchmark_sha256": hashlib.sha256(
            (args.output / "benchmark.npz").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
