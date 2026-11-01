# MachineLearning
460-4162/01 | Winter 2026/27

## Banana ripeness baseline

This repository contains a pure-PyTorch Milestone 1 baseline. It deliberately
uses a custom `Dataset`, `DataLoader`, `nn.Module`, explicit forward/loss/
backward/optimizer steps, and a tiny-slice overfit sanity check.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the dataset (Kaggle authentication may be required):

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('shahriar26s/banana-ripeness-classification-dataset'))"
```

Run the five-class baseline:

```bash
python train.py --data /Users/jakub/.cache/kagglehub/datasets/shahriar26s/banana-ripeness-classification-dataset/versions/1
```

The source dataset has four classes: `unripe`, `ripe`, `overripe`, and
`rotten`. Five-class mode deterministically splits `ripe` into
`partially_ripe` and `ripe` using mean image hue. These are pseudo-labels and
must be treated as an experiment; use `--four-classes` for the original labels.
The script discovers class directories recursively, so it supports datasets
with either a single root or separate train/validation/test folders. For this
Milestone 1 baseline it creates a reproducible 80/20 split.
