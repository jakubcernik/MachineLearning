# MachineLearning
460-4162/01 | Winter 2026/27

## Banana ripeness baseline

This repository contains a pure-PyTorch Milestone 1 baseline. It uses the
dataset's real folder labels and deliberately implements a custom `Dataset`,
`DataLoader`, `nn.Module`, explicit forward/loss/backward/optimizer steps, and
a tiny-slice overfit sanity check. No class is split or pseudo-labeled.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Set the Roboflow API key in the shell; do not put it in source code:

```bash
export ROBOFLOW_API_KEY="35UXIlutAa1tF0GxX1ee"
python download_dataset.py
```

Run training:

```bash
python train.py --data roboflow
```

The loader uses `train`, `valid`/`validation`, and `test` directories when
provided by Roboflow. If no validation directory exists, it creates a
reproducible 80/20 split from training data. Class names are discovered from
the dataset folders, so the script supports the dataset's actual number of
classes without hardcoded labels.

The script writes two visualizations:

- `outputs/batch_preview.png` shows a transformed input batch with dataset
  labels. It is the required data-pipeline inspection, not a prediction.
- `outputs/prediction_preview.png` shows the trained model's true label versus
  predicted label on validation images.

Training also writes:

- `outputs/best_model.pt`, selected by lowest validation loss;
- `outputs/metrics.csv`, containing loss and accuracy for every epoch;
- `outputs/validation_confusion_matrix.png`;
- `outputs/test_confusion_matrix.png` when a test split is provided.

The console reports class counts, per-class precision/recall/F1, the selected
best epoch, and test metrics. Early stopping is enabled by default after three
epochs without validation-loss improvement; change it with `--patience`.

The API key previously pasted into chat should be revoked and regenerated in
Roboflow before use.
