"""Prepare the Musical Instruments catalog and links from a gzip JSONL file."""

import argparse
import gzip
from itertools import islice
import json
from pathlib import Path

import pandas as pd


def prepare(raw: Path, output: Path, max_rows: int | None) -> None:
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    with gzip.open(raw, "rt", encoding="utf-8") as source:
        rows = []
        for line in islice(source, max_rows):
            row = json.loads(line)
            categories = row["category"]
            rows.append({
                "item_id": row["asin"],
                "title": row.get("title"),
                "description": row.get("description"),
                "category_path": " > ".join(categories),
                "category": categories[-1] if categories else None,
                "buy": row.get("also_buy") or [],
                "view": row["also_view"],
            })
    if not rows or (max_rows is not None and len(rows) != max_rows):
        raise ValueError("The input has fewer records than requested")
    catalog = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    items = catalog.drop(columns=["buy", "view"])
    categories = items.set_index("item_id")["category"]
    pieces = []
    for relation in ("buy", "view"):
        pairs = catalog[["item_id", relation]].explode(relation)
        pairs.columns = ["source_id", "target_id"]
        pairs = pairs.dropna().drop_duplicates()
        pairs = pairs.loc[pairs["source_id"].ne(pairs["target_id"]) & pairs["target_id"].isin(categories.index)]
        pieces.append(pairs.assign(relation=relation))
    pairs = pd.concat(pieces, ignore_index=True)
    pairs["source_category"] = pairs["source_id"].map(categories)
    pairs["target_category"] = pairs["target_id"].map(categories)
    pairs = pairs.dropna(subset=["source_category", "target_category"])
    output.mkdir(parents=True, exist_ok=False)
    items.to_csv(output / "items.csv.gz", index=False)
    pairs.to_csv(output / "pairs.csv.gz", index=False)
    print(f"{len(rows):,} metadata rows, {len(items):,} distinct items, {len(pairs):,} directed links")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.max_rows)
