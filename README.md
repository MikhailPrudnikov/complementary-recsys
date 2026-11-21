# Shared-factors model for complementary product recs

A research project on recommending complementary products, inspired by [Complementary Two-Tower at Allegro](https://arxiv.org/abs/2508.03702).

Given a source product and a target category, we rank the top ten products in that category. We investigate whether jointly viewed products provide a useful training signal when jointly purchased product links are scarce.

**Status:** Amazon and REES46 pilot results are available in the [interim report](docs/interim-report.pdf). The CLI and notebook run the factorization experiments and the local Buy-only, Mixed, and Shared controls. Pretrain and Transformer are planned.

## Research questions

1. Does joint learning with separate purchase/view heads outperform purchase-only training?
2. Does separating the two relations work better than mixing their links?
3. Does joint learning outperform view pretraining followed by purchase fine-tuning?
4. Can Tucker 3D outperform popularity and purchase-only baselines?
5. Does the advantage of Shared and Tucker increase as purchase links become scarce?
6. Can Tucker match Shared's recommendation quality with fewer parameters and less training time?

We will also examine how these findings vary across datasets, compare Tucker training with HOSVD, and investigate user context through a 4D model.

## Baselines

| Method | Description |
| --- | --- |
| MostPopular | Rank products within the target category by incoming purchase-link counts in the training graph. |
| Buy-only | A two-tower model trained on purchase links only. |
| Transformer (planned) | A pair-scoring encoder conditioned on one of two learned relation tokens: `[BUY]` or `[VIEW]`. Recommendations use `[BUY]`. |

## Methods under study

| Method | Description |
| --- | --- |
| Mixed | Train on purchase and view links without distinguishing their relation types. |
| Shared | Use a shared product encoder with separate query and candidate heads for each relation. |
| Pretrain → Buy (planned) | Pretrain on view links, then fine-tune on purchase links. |
| Tucker 3D | Model source product × candidate product × relation, sharing factors between the product axes. |
| HOSVD | Study truncated higher-order SVD as an alternative way to compute Tucker factors, accounting for differences in constraints and training objectives. |
| Tucker 4D | Add a user axis on datasets with user-level event logs and compare with 3D on the same evaluation tasks. |

Shared optimizes `L_buy + λ L_view`, with λ selected on validation data. Its purchase branch is used for recommendation. Neural models and gradient-trained Tucker models use binary cross-entropy on observed and sampled unobserved links.

## Data

We start with [Amazon Review Data 2018: Musical Instruments](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), using product titles, categories, `also_buy`, and `also_view`. The interim Amazon study uses a fixed pair split and a catalog of 20,000 products.

Preparation and audit notebooks also cover [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), [REES46 Cosmetics and REES46 Electronics](https://rees46.com/en/datasets). The interim report includes a transfer check on Electronics and a personalized 3D/4D comparison on Cosmetics.

Amazon metadata provides static product-link graphs, without user identities or event timestamps. The 4D experiments therefore require additional event-log datasets. Purchase links are a proxy for complementarity, not a guarantee of product compatibility.

## Evaluation

- On Amazon, compare models using nested 100%, 50%, and 25% subsets of purchase training links, with fixed view links and test data.
- Group each product pair with its reverse when splitting Amazon links; remove held-out pairs from both training relations.
- Split event logs chronologically, keeping baskets and sessions intact.
- Use common candidates and filtering rules within each comparison. Evaluate known products and, for 4D, known users; report coverage.
- Report NDCG@10, Recall@10, HitRate@10, parameter counts, training time, and inference time. HitRate@10 is the share of evaluated queries with at least one relevant target among the top ten. Use three initializations for gradient-trained models and comparable tuning budgets.

Tucker learns from product identities and relations, while neural models can also use available product features. Comparisons therefore assess the complete approaches. Results are compared within each dataset.

## Work areas

| Area | Responsibilities |
| --- | --- |
| Two-tower models | Shared encoder, Shared, Pretrain, and Transformer |
| Data and evaluation | Dataset discovery and auditing, splits, metrics, MostPopular, and Buy-only |
| Tensor models | Tucker 3D/4D, HOSVD, Mixed, and computational cost analysis |

Experiment runs, analysis, report writing, and cross-review are shared across the team.

## Plan

| Date (2026) | Milestone |
| --- | --- |
| October 22 | Proposal and defense |
| November 21 | Interim report |
| December 3 | Sparsity experiments, additional datasets, HOSVD, 4D, and repeated runs |
| December 9 | Final report |
| December 10–11 | Project defense |

## References

- [Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com](https://arxiv.org/abs/2508.03702).
- [Training a Tucker Model With Shared Factors](https://proceedings.mlr.press/v238/peshekhonov24a.html).
- [Kolda and Bader: Tensor Decompositions and Applications](https://doi.org/10.1137/07070111X).

The shared-factor Tucker model is inspired by the cited work; reproducing its Riemannian optimizer is outside the current plan.

## Running the factorization experiments

Use Python 3.11 or newer. From the repository root, install the project with
`python -m pip install -e .`. For Amazon, run the Amazon sections of
[dataset_preparation.ipynb](datasets/dataset_preparation.ipynb). The commands
below freeze one copy of the split and run all static models on validation.

```bash
complementary-recsys split-amazon --processed data/processed/amazon \
  --output data/splits/amazon-v1 --seed 42
complementary-recsys run --split data/splits/amazon-v1 \
  --config configs/amazon_pilot.json --output runs/amazon-validation \
  --revision "$(git rev-parse HEAD)"
```

The run writes `metrics.csv`, per-query `recommendations.csv.gz`, the exact
catalog, and a manifest with split and source fingerprints, config and runtime versions.
Use a new output directory for each run. Choose model settings on validation
before running the finalized configuration with `--partition test`.
For the sparsity experiment, copy the config and set `buy_fraction` to
`0.5` or `0.25` with the same `subsample_seed`. The runner keeps the
evaluation catalog and known-buy exclusions fixed. Each epoch uses all
retained buys and a seeded view sample with a fixed size across buy fractions.

For a personalized comparison on REES46 Electronics, first run its preparation
and chronological splitting sections, then run:

```bash
complementary-recsys run --split data/splits/seed_42/rees46_electronics \
  --config configs/rees46_personalization.json --output runs/electronics-validation
```

The temporal evaluator uses categories inferred from training pairs and reports
coverage after filtering to eligible users, items and categories. The 3D and 4D
models receive the same user-tagged pairs. The 3D scorer ignores user IDs and
the 4D scorer conditions on the query user. MostPopular and graph paths have
no learned parameters. HOSVD reports the number of scalar factor entries.
MIG-TF uses the paper's Lorentz score with a frozen pretrained Tucker base.
The training protocol is adapted to this project with sampled BCE, curvature
fixed at −1, and no paper-specific dropout or reciprocal-edge augmentation.

For Kaggle, [amazon_experiments.ipynb](notebooks/amazon_experiments.ipynb)
uses the same package and config. Point it at a pinned source snapshot and
a pinned split with its manifest. Set `RECSYS_CODE_REVISION` to the source
snapshot identifier. For a CLI run on a GPU, add `--device cuda`.

### Reproduce the interim experiments

Build `data/splits/amazon-v1` from the [official Musical Instruments metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz)
with the Amazon sections of the preparation notebook, or use the local script:

```bash
python scripts/prepare_amazon_study.py \
  --raw data/raw/amazon/meta_Musical_Instruments.json.gz --output data/processed/amazon
complementary-recsys split-amazon --processed data/processed/amazon \
  --output data/splits/amazon-v1 --seed 42
```

Then run:

```bash
python scripts/run_study.py --split data/splits/amazon-v1 \
  --config configs/amazon_train_only.json --output runs/amazon-train-only-reproduction --fixed-settings
```

The corrected Amazon study reuses fixed ranks and weights selected on validation in the initial study. That selection considered ranks 16 and 32. Controls within one family
share the rank with the highest mean validation NDCG@10 for that family.
It also selects a view weight from 0.25, 0.5 and 1 for the joint Tucker,
Shared and graph comparisons. The runner records the fixed settings in `selected_protocol.json` and evaluates
100%, 50% and 25% of purchases with seeds 42, 43 and 44. The two-tower
Buy-only, Mixed and Shared rows are local matched controls with product IDs,
category embeddings and hashed title tokens. In the current Mixed, each epoch
merges buys with a fresh view sample, and all known training views are protected
from negative sampling. Mixed uses the requested target category at scoring.
The historical Shared and Buy-only controls keep their original training policy.
The project's own gradient models use train-only negative sampling.
See the [correction study](docs/amazon-training-fixes.md) for the train-only config
and the separate comparison at the original hyperparameters.
For an existing study, `python scripts/backfill_hit_rate.py <study-directory>`
adds HitRate@10 from its saved rankings while preserving the original training manifests.

For the 4D study, download the original [REES46 Cosmetics CSVs](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop)
to `data/raw/rees46_cosmetics`; then install `duckdb` and run:

```bash
python scripts/prepare_cosmetics_temporal.py --raw data/raw/rees46_cosmetics \
  --output data/splits/cosmetics-active-v1 --database /tmp/cosmetics-study.duckdb \
  --max-users 3000
python scripts/run_study.py --split data/splits/cosmetics-active-v1 \
  --config configs/rees46_personalization.json --output runs/rees46-cosmetics-personalization-reproduction
```

The cohort is selected from training sessions after the global time split.
The split report records cutoff times, cohort rules and pair counts. Tucker
3D and 4D share a product rank selected from 8 and 16 on validation. Both
rank the same test queries and the same candidates.

The [report inputs](docs/results/) contain the corrected Amazon results, the historical Amazon summaries, and the REES46 summaries. The [historical reproduction archive](docs/historical-reproduction.zip) contains the original code and settings for the Amazon controls. Install `reportlab` to render the same PDF:

```bash
python scripts/build_interim_report.py --amazon docs/results/amazon-train-only \
  --historical-amazon docs/results/amazon-historical \
  --temporal docs/results/rees46-cosmetics-personalization --transfer docs/results/rees46-electronics-transfer \
  --examples docs/results/amazon-historical/reviewed_examples.json \
  --output docs/interim-report.pdf --full-amazon
```
