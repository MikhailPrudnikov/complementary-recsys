"""Replay the validation runs recorded in validation_runs.csv."""
import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPTIONS = (
    "model", "seed", "buy_fraction", "dim", "steps", "pretrain_steps",
    "batch_size", "negatives", "view_weight", "lr", "eval_every", "patience",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["amazon-expanded-v2", "electronics-expanded-v2", "electronics-id-v2"])
    parser.add_argument("--run", help="Exact run name in the CSV")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    with (HERE / "validation_runs.csv").open() as source:
        rows = [
            row for row in csv.DictReader(source)
            if (args.dataset is None or row["dataset"] == args.dataset)
            and (args.run is None or row["run"] == args.run)
        ]
    if not rows:
        parser.error("No matching run in validation_runs.csv")
    for dataset in sorted({row["dataset"] for row in rows}):
        artifact = args.data_root / dataset / "benchmark.npz"
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if {r["benchmark_sha256"] for r in rows if r["dataset"] == dataset} != {actual}:
            parser.error(f"Input checksum mismatch for {dataset}")
    for row in rows:
        data = (args.data_root / row["dataset"]).resolve()
        output = (args.output / row["dataset"] / row["run"]).resolve()
        if output.exists():
            parser.error(f"Output already exists: {output}. Select a new output directory.")
        cmd = [sys.executable, "-m", "recsys.train", "--data", str(data),
               "--output", str(output), "--device", args.device]
        for option in OPTIONS:
            cmd.extend(["--" + option.replace("_", "-"), row[option]])
        if row["shared_negative_pool"] == "True":
            cmd.append("--shared-negative-pool")
        subprocess.run(cmd, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
