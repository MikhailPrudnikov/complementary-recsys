"""Reproducible pilot trainer. Test evaluation is an explicit opt-in."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .models import ContentEncoder, PairTransformer, TwoTower


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def density_subset(edges, fraction, seed=1729):
    keys = sorted({tuple(sorted(map(int, p))) for p in edges})
    order = np.random.default_rng(seed).permutation(len(keys))
    keep = {keys[i] for i in order[: max(1, int(len(keys) * fraction))]}
    return np.array(
        [p for p in edges if tuple(sorted(map(int, p))) in keep], dtype=np.int64
    ).reshape(-1, 2)


class NegativeSampler:
    def __init__(self, categories, positives, holdout):
        self.categories = categories
        self.pools = {c: np.flatnonzero(categories == c) for c in np.unique(categories)}
        self.blocked = defaultdict(set)
        for a, b in positives:
            self.blocked[int(a)].add(int(b))
        for a, b in holdout:
            self.blocked[int(a)].add(int(b))
            self.blocked[int(b)].add(int(a))
        self.cache = {}

    def allowed(self, source, category):
        key = (int(source), int(category))
        if key not in self.cache:
            forbidden = self.blocked[key[0]] | {key[0]}
            self.cache[key] = np.array(
                [j for j in self.pools[category] if int(j) not in forbidden],
                dtype=np.int64,
            )
        return self.cache[key]

    def batch(self, edges, rng, size, n_negative):
        selected, negatives = [], []
        attempts = 0
        while len(selected) < size:
            attempts += 1
            if attempts > size * 50:
                raise ValueError("Too few eligible queries for negative sampling")
            a, b = edges[rng.integers(len(edges))]
            pool = self.allowed(a, self.categories[b])
            if not len(pool):
                continue
            selected.append((a, b))
            negatives.append(
                rng.choice(pool, size=n_negative, replace=len(pool) < n_negative)
            )
        return np.asarray(selected), np.asarray(negatives)


def pair_loss(model, edges, sampler, rng, batch_size, negatives, relation, device):
    pairs, neg = sampler.batch(edges, rng, batch_size, negatives)
    source = torch.as_tensor(pairs[:, 0], device=device)
    candidates = torch.as_tensor(np.column_stack([pairs[:, 1], neg]), device=device)
    category = torch.as_tensor(sampler.categories[pairs[:, 1]], device=device)
    if isinstance(model, TwoTower):
        q = model.query(source, category, relation)
        c = model.candidate(candidates.reshape(-1), relation).reshape(
            batch_size, negatives + 1, -1
        )
        branch = model.branch(relation)
        logits = (q[:, None, :] * c).sum(-1) * model.log_scale[branch].clamp(
            -2, 4.6
        ).exp() + model.bias[branch]
    else:
        logits = model(
            source[:, None].expand_as(candidates).reshape(-1),
            candidates.reshape(-1),
            category[:, None].expand_as(candidates).reshape(-1),
            relation,
        ).reshape(batch_size, -1)
    # Class-balanced sampled BCE: increasing K does not silently increase negative weight.
    return 0.5 * (F.softplus(-logits[:, 0]).mean() + F.softplus(logits[:, 1:]).mean())


def rank_metrics(ranked, positives, k=10):
    hits = [int(int(i) in positives) for i in ranked[:k]]
    recall = sum(hits) / len(positives)
    dcg = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
    ideal = sum(1 / math.log2(r + 2) for r in range(min(k, len(positives))))
    return dcg / ideal, recall


def evaluation_queries(edges, categories, max_queries=0):
    groups = defaultdict(set)
    for a, b in edges:
        groups[(int(a), int(categories[b]))].add(int(b))
    keys = sorted(groups)
    if max_queries and len(keys) > max_queries:
        selected = np.random.default_rng(9001).choice(
            len(keys), max_queries, replace=False
        )
        keys = [keys[i] for i in sorted(selected)]
    return [(a, c, groups[(a, c)]) for a, c in keys]


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    if device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def evaluate(
    model, edges, categories, full_train_buy, device, max_queries=0, popularity=None
):
    if model is not None:
        model.eval()
    queries = evaluation_queries(edges, categories, max_queries)
    by_category = defaultdict(list)
    seen = defaultdict(set)
    for a, b in full_train_buy:
        seen[int(a)].add(int(b))
    for q in queries:
        by_category[q[1]].append(q)
    synchronize(device)
    start = time.perf_counter()
    cache = None
    if isinstance(model, TwoTower):
        cache = torch.cat(
            [
                model.candidate(
                    torch.arange(a, min(a + 1024, len(categories)), device=device)
                )
                for a in range(0, len(categories), 1024)
            ]
        )
    rows = []
    for category, group in sorted(by_category.items()):
        pool = np.flatnonzero(categories == category)
        p = torch.as_tensor(pool, device=device)
        for offset in range(0, len(group), 32):
            chunk = group[offset : offset + 32]
            if popularity is not None:
                scores = np.tile(popularity[pool], (len(chunk), 1)).astype(float)
            elif cache is not None:
                source = torch.tensor([q[0] for q in chunk], device=device)
                cats = torch.full_like(source, category)
                scores = (model.query(source, cats) @ cache[p].T).cpu().numpy()
            else:
                # Pair Transformer must score the complete eligible category, in bounded blocks.
                scores = np.empty((len(chunk), len(pool)), dtype=np.float32)
                for row, (a, c, _) in enumerate(chunk):
                    for begin in range(0, len(pool), 512):
                        target = p[begin : begin + 512]
                        values = model(
                            torch.full_like(target, a),
                            target,
                            torch.full_like(target, c),
                            0,
                        )
                        scores[row, begin : begin + len(target)] = values.cpu().numpy()
            for row, (source, cat, positives) in enumerate(chunk):
                blocked = seen[source] | {source}
                if positives & blocked:
                    raise ValueError(
                        "Evaluation positives overlap masked training edges; choose an explicit repeat-event protocol"
                    )
                mask = np.array([int(v) in blocked for v in pool])
                s = scores[row].copy()
                s[mask] = -np.inf
                eligible = np.flatnonzero(~mask)
                # Stable item-index tie break, shared by all methods.
                order = eligible[np.lexsort((pool[eligible], -s[eligible]))][:10]
                ndcg, recall = rank_metrics(pool[order], positives)
                rows.append(
                    {
                        "source": source,
                        "category": cat,
                        "ndcg10": ndcg,
                        "recall10": recall,
                        "n_positives": len(positives),
                        "n_candidates": len(eligible),
                    }
                )
    synchronize(device)
    return {
        "ndcg10": float(np.mean([x["ndcg10"] for x in rows])),
        "recall10": float(np.mean([x["recall10"] for x in rows])),
        "n_queries": len(rows),
        "seconds_including_candidate_encoding": time.perf_counter() - start,
        "mean_candidates": float(np.mean([x["n_candidates"] for x in rows])),
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["buy", "shared", "mixed", "pretrain", "transformer", "popular"],
        default="shared",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dim", type=int, choices=[128, 256], default=128)
    parser.add_argument("--shared-negative-pool", action="store_true", help="Use the union mask for both tasks when training on both relations")
    parser.add_argument("--pretrain-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--view-weight", type=float, default=0.3)
    parser.add_argument(
        "--buy-fraction", type=float, choices=[1.0, 0.5, 0.25], default=1.0
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument(
        "--validation-queries",
        type=int,
        default=0,
        help="0 means all; nonzero is a labelled development subset",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output is not empty; use a new run directory")
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.negatives < 1
        or args.eval_every < 1
    ):
        raise ValueError(
            "Steps, batch size, negatives and evaluation interval must be positive"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    arrays = dict(np.load(args.data / "benchmark.npz", allow_pickle=False))
    manifest = json.loads((args.data / "manifest.json").read_text())
    if digest(args.data / "benchmark.npz") != manifest["benchmark_sha256"]:
        raise ValueError("Benchmark checksum mismatch")
    categories = arrays["categories"]
    full_buy = arrays["train_buy"]
    buy = density_subset(full_buy, args.buy_fraction)
    view = arrays["train_view"]
    # Removed buy positives remain unknown, not newly-created negative labels in sparsity runs.
    # This frozen benchmark mask is shared by methods and disclosed in the protocol.
    buy_sampler = NegativeSampler(categories, full_buy, arrays["holdout"])
    view_sampler = NegativeSampler(categories, view, arrays["holdout"])
    mixed_sampler = NegativeSampler(
        categories,
        np.unique(np.concatenate([full_buy, view]), axis=0),
        arrays["holdout"],
    )
    if args.shared_negative_pool and args.model in {"mixed", "shared", "pretrain", "transformer"}:
        buy_sampler = view_sampler = mixed_sampler
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        device=device,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        benchmark_sha256=manifest["benchmark_sha256"],
        train_buy=len(buy),
        train_view=len(view),
        loss="0.5*(mean_positive_BCE+mean_negative_BCE) per task",
        sampling="uniform within target category, relation-positive and holdout exclusion",
        evaluation="exact category ranking; fixed full-buy-train mask; macro query mean",
        validation_subset=bool(args.validation_queries),
    )
    config["dependency_versions"] = {
        name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas"]
    }
    config["source_sha256"] = {
        p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    if args.model == "popular":
        pop = np.bincount(buy[:, 1], minlength=len(categories))
        result, rows = evaluate(
            None,
            arrays["validation_buy"],
            categories,
            full_buy,
            device,
            args.validation_queries,
            pop,
        )
        (args.output / "validation.json").write_text(json.dumps(result, indent=2))
        print(result, flush=True)
        return
    encoder = ContentEncoder(
        arrays["tokens"], categories, manifest["vocab_size"], manifest["n_categories"], args.dim
    )
    model = (
        PairTransformer(encoder, args.dim)
        if args.model == "transformer"
        else TwoTower(encoder, args.model == "shared", args.dim)
    ).to(device)
    config["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    config["frozen_parameters"] = 0
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history_path = args.output / "history.jsonl"
    start = time.perf_counter()
    train_seconds = 0.0
    best = -1.0
    stale = 0
    phases = (
        [("view_pretrain", args.pretrain_steps), ("buy_finetune", args.steps)]
        if args.model == "pretrain"
        else [("train", args.steps)]
    )
    last_step = 0
    for phase, steps in phases:
        if phase == "buy_finetune":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr / 3, weight_decay=1e-4
            )
        for step in range(1, steps + 1):
            model.train()
            synchronize(device)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if phase == "view_pretrain":
                loss = pair_loss(
                    model,
                    view,
                    view_sampler,
                    rng,
                    args.batch_size,
                    args.negatives,
                    0,
                    device,
                )
                buy_loss = None
                view_loss = loss
            elif args.model == "mixed":
                buy_loss = pair_loss(
                    model,
                    buy,
                    mixed_sampler,
                    rng,
                    args.batch_size,
                    args.negatives,
                    0,
                    device,
                )
                view_loss = pair_loss(
                    model,
                    view,
                    mixed_sampler,
                    rng,
                    args.batch_size,
                    args.negatives,
                    0,
                    device,
                )
                loss = buy_loss + args.view_weight * view_loss
            else:
                buy_loss = pair_loss(
                    model,
                    buy,
                    buy_sampler,
                    rng,
                    args.batch_size,
                    args.negatives,
                    0,
                    device,
                )
                loss = buy_loss
                view_loss = None
                if args.model in ["shared", "transformer"] and args.view_weight > 0:
                    view_loss = pair_loss(
                        model,
                        view,
                        view_sampler,
                        rng,
                        args.batch_size,
                        args.negatives,
                        1,
                        device,
                    )
                    loss = loss + args.view_weight * view_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            synchronize(device)
            train_seconds += time.perf_counter() - tick
            last_step = step
            record = {
                "phase": phase,
                "step": step,
                "loss": float(loss.detach()),
                "buy_loss": float(buy_loss.detach()) if buy_loss is not None else None,
                "view_loss": float(view_loss.detach())
                if view_loss is not None
                else None,
                "train_seconds": train_seconds,
            }
            if step == 1 or step % 25 == 0:
                print(json.dumps(record), flush=True)
            if phase != "view_pretrain" and (
                step % args.eval_every == 0 or step == steps
            ):
                metrics, rows = evaluate(
                    model,
                    arrays["validation_buy"],
                    categories,
                    full_buy,
                    device,
                    args.validation_queries,
                )
                record["validation"] = metrics
                print(
                    json.dumps({"phase": phase, "step": step, "validation": metrics}),
                    flush=True,
                )
                if metrics["ndcg10"] > best:
                    best = metrics["ndcg10"]
                    stale = 0
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": config,
                            "phase": phase,
                            "step": step,
                            "validation": metrics,
                            "numpy_rng": rng.bit_generator.state,
                            "torch_rng": torch.get_rng_state(),
                        },
                        args.output / "best.pt",
                    )
                    (args.output / "validation.json").write_text(
                        json.dumps(metrics, indent=2)
                    )
                    (args.output / "validation_per_query.json").write_text(
                        json.dumps(rows)
                    )
                else:
                    stale += 1
            with history_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if phase != "view_pretrain" and stale >= args.patience:
                break
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "phase": phase,
            "step": last_step,
        },
        args.output / "last.pt",
    )
    if args.evaluate_test:
        saved = torch.load(
            args.output / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(saved["model"])
        metrics, rows = evaluate(
            model, arrays["test_buy"], categories, full_buy, device, 0
        )
        (args.output / "test.json").write_text(json.dumps(metrics, indent=2))
        (args.output / "test_per_query.json").write_text(json.dumps(rows))
    summary = {
        "best_validation_ndcg10": best,
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - start,
        "trainable_parameters": config["trainable_parameters"],
        "test_evaluated": args.evaluate_test,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
