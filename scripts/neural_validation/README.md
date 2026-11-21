# Neural validation experiments

These are the expanded validation experiments on Amazon Musical Instruments and REES46 Electronics. The Amazon test study in the repository root uses a different catalog and protocol. The CSV here records the original settings and validation metrics for each of 97 runs, including popularity controls and sensitivity checks. Each row names its dataset, random seed and benchmark fingerprint. Each reported neural score uses the checkpoint selected by validation NDCG@10. The test option is disabled in the replay command.

## Environment and raw inputs

From this directory, use Python 3.12. The original environment used Python 3.12.14, NumPy 2.5.5, pandas 3.1.1 and PyTorch 2.15.0 on Apple MPS. CPU and CUDA use the same protocol; small ranking differences across backends are possible.

```bash
python -m pip install numpy==2.5.5 pandas==3.1.1 torch==2.15.0
python -m recsys.prepare --raw data/raw/meta_Musical_Instruments.json.gz \
  --output data/amazon-expanded-v2 --max-items 100000 --max-vocab 32000
```

The Amazon command downloads [the original metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz) if the file is absent. Download the [REES46 Electronics event file](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store), then pass its CSV or its single-file ZIP to both preparation commands. The ZIP used in the study had SHA-256 `f85ec03b665a10211fc99c85a0b1ab1f7d437fde3352c581ae395630647b7ea6`.

```bash
python prepare_electronics.py --raw data/raw/electronics.zip --output data/electronics-expanded-v2
python prepare_electronics.py --raw data/raw/electronics.zip --output data/electronics-id-v2 --item-id
```

## Reproduce the table

```bash
python replay.py --data-root data --output runs/example --dataset electronics-id-v2 --run mixed-0.25-42
python replay.py --data-root data --output runs/full
```

Use a fresh output path. Add `--device mps` to select the original backend explicitly. The runner checks the SHA-256 of each prepared benchmark. Take the mean of `validation.json` values over the matching training seeds (42, 43, 44) for each model and purchase fraction. Popularity uses one deterministic ranking; Pretrain and Transformer on expanded catalogs use seed 42 only. Validation NDCG@10 is the checkpoint selection metric. All candidates in the target category are ranked. The fractions subsample unordered purchase pairs with a fixed order; the full training purchase mask is kept for evaluation. Electronics uses newly observed cross-category purchase pairs at validation. The ID variant assigns a token to each item present in a training edge; all other items share an unknown token.
