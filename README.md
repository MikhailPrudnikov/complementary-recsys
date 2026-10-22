# Shared-factors model for complementary product recs

A research project on recommending complementary products, inspired by [Complementary Two-Tower at Allegro](https://arxiv.org/abs/2508.03702).

Given a source product and a target category, we rank the top ten products in that category. We investigate whether jointly viewed products provide a useful training signal when jointly purchased product links are scarce.

**Status:** the proposal has been presented and revised after feedback. Implementation and experiments are planned; no model results are available yet.

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
| Transformer | A pair-scoring encoder conditioned on one of two learned relation tokens: `[BUY]` or `[VIEW]`. Recommendations use `[BUY]`. |

## Methods under study

| Method | Description |
| --- | --- |
| Mixed | Train on purchase and view links without distinguishing their relation types. |
| Shared | Use a shared product encoder with separate query and candidate heads for each relation. |
| Pretrain → Buy | Pretrain on view links, then fine-tune on purchase links. |
| Tucker 3D | Model source product × candidate product × relation, sharing factors between the product axes. |
| HOSVD | Study truncated higher-order SVD as an alternative way to compute Tucker factors, accounting for differences in constraints and training objectives. |
| Tucker 4D | Add a user axis on datasets with user-level event logs and compare with 3D on the same evaluation tasks. |

Shared optimizes `L_buy + λ L_view`, with λ selected on validation data. Its purchase branch is used for recommendation. Neural models and gradient-trained Tucker models use binary cross-entropy on observed and sampled unobserved links.

## Data

We start with [Amazon Review Data 2018: Musical Instruments](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), using product titles, categories, `also_buy`, and `also_view`. The metadata has been structurally audited; final category selection and evaluation splits are pending.

We will identify and validate at least three additional datasets from different retail domains. Current candidates are [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), [REES46 Cosmetics and REES46 Electronics](https://rees46.com/en/datasets). Their suitability will be checked before experiments.

Amazon metadata provides static product-link graphs, without user identities or event timestamps. The 4D experiments therefore require additional event-log datasets. Purchase links are a proxy for complementarity, not a guarantee of product compatibility.

## Evaluation

- On Amazon, compare models using nested 100%, 50%, and 25% subsets of purchase training links, with fixed view links and test data.
- Group each product pair with its reverse when splitting Amazon links; remove held-out pairs from both training relations.
- Split event logs chronologically, keeping baskets and sessions intact.
- Use common candidates and filtering rules within each comparison. Evaluate known products and, for 4D, known users; report coverage.
- Report NDCG@10, Recall@10, parameter counts, training time, and inference time. Use three initializations for gradient-trained models and comparable tuning budgets.

Tucker learns from product identities and relations, while neural models can also use available product features. Comparisons therefore assess the complete approaches. Results are compared within each dataset.

## Team

| Member | Responsibilities |
| --- | --- |
| Mikhail Prudnikov | Shared encoder, Shared, Pretrain, and Transformer |
| Mikhail Karachurin | Dataset discovery and auditing, splits, metrics, MostPopular, and Buy-only |
| Filipp Nikulenko | Tucker 3D/4D, HOSVD, Mixed, and computational cost analysis |

Experiment runs, analysis, report writing, and cross-review are shared across the team.

## Plan

| Date (2026) | Milestone |
| --- | --- |
| October 22 | Proposal and defense |
| November 19 | Additional dataset audit and initial comparison |
| December 3 | Sparsity experiments, additional datasets, HOSVD, 4D, and repeated runs |
| December 9 | Final report |
| December 10–11 | Project defense |

We plan a pilot on one Kaggle GPU and catalogs of up to 20,000 products per dataset. The pilot will determine runtime and memory requirements. If needed, we will reduce catalog sizes, factor ranks, and tuning grids while retaining the research directions.

Reproduction instructions will be added with the implementation.

## References

- [Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com](https://arxiv.org/abs/2508.03702).
- [Training a Tucker Model With Shared Factors](https://proceedings.mlr.press/v238/peshekhonov24a.html).
- [Kolda and Bader: Tensor Decompositions and Applications](https://doi.org/10.1137/07070111X).

The shared-factor Tucker model is inspired by the cited work; reproducing its Riemannian optimizer is outside the current plan.
