"""Pure-PyTorch baseline for a folder-structured banana dataset."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def image_to_tensor(image: Image.Image, size: int, training: bool) -> torch.Tensor:
    """Resize, optionally flip, and convert HWC uint8 pixels to CHW floats."""
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    if training and random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - 0.5) / 0.5


def class_directories(directory: Path) -> list[Path]:
    """Return directories containing images, sorted for stable class indices."""
    return sorted(
        path for path in directory.iterdir()
        if path.is_dir() and any(
            child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
            for child in path.iterdir()
        )
    )


def samples_in_directory(
    directory: Path, class_to_index: dict[str, int]
) -> list[tuple[Path, int, str]]:
    samples = []
    for class_directory in class_directories(directory):
        class_name = class_directory.name
        if class_name not in class_to_index:
            raise ValueError(f"Unknown class {class_name!r} in {directory}")
        samples.extend(
            (path, class_to_index[class_name], class_name)
            for path in sorted(class_directory.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return samples


def collect_samples(
    root: Path,
) -> tuple[
    list[tuple[Path, int, str]],
    list[tuple[Path, int, str]],
    list[tuple[Path, int, str]],
    list[str],
]:
    """Read real class labels and provided train/validation/test directories."""
    split_directories = {
        "train": next((path for path in root.rglob("train") if path.is_dir()), None),
        "validation": next(
            (path for path in root.rglob("valid") if path.is_dir()), None
        ) or next((path for path in root.rglob("validation") if path.is_dir()), None),
        "test": next((path for path in root.rglob("test") if path.is_dir()), None),
    }
    if split_directories["train"] is None:
        split_directories = {"train": root, "validation": None, "test": None}

    discovered_classes = {
        class_directory.name
        for directory in split_directories.values()
        if directory is not None
        for class_directory in class_directories(directory)
    }
    if len(discovered_classes) < 2:
        raise ValueError(f"Expected at least two class directories below {root}")
    class_names = sorted(discovered_classes)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_samples = samples_in_directory(split_directories["train"], class_to_index)
    validation_samples = (
        samples_in_directory(split_directories["validation"], class_to_index)
        if split_directories["validation"] is not None
        else []
    )
    test_samples = (
        samples_in_directory(split_directories["test"], class_to_index)
        if split_directories["test"] is not None
        else []
    )
    if not train_samples:
        raise ValueError(f"No training images found below {root}")
    if not validation_samples:
        random.shuffle(train_samples)
        split_index = max(1, int(len(train_samples) * 0.8))
        validation_samples = train_samples[split_index:]
        train_samples = train_samples[:split_index]
    return train_samples, validation_samples, test_samples, class_names


class BananaDataset(Dataset[tuple[torch.Tensor, int]]):
    """Custom Dataset demonstrating explicit __len__ and __getitem__."""

    def __init__(
        self,
        samples: list[tuple[Path, int, str]],
        image_size: int = 128,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label, _ = self.samples[index]
        with Image.open(path) as image:
            tensor = image_to_tensor(image, self.image_size, self.training)
        return tensor, label


class BaselineCNN(nn.Module):
    """Small CNN whose tensor transitions are easy to explain at the defense."""

    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, number_of_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, start_dim=1))


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    number_of_classes: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate loss, accuracy, predictions, and labels without gradients."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = loss_function(predictions, labels)
        predicted_labels = predictions.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)
        all_predictions.extend(predicted_labels.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return (
        total_loss / total,
        correct / total,
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_labels, dtype=np.int64),
    )


def class_metrics(
    predictions: np.ndarray, labels: np.ndarray, number_of_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate per-class precision, recall, and F1 without sklearn."""
    confusion = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1
    precision = np.divide(
        np.diag(confusion), confusion.sum(axis=0),
        out=np.zeros(number_of_classes, dtype=float),
        where=confusion.sum(axis=0) != 0,
    )
    recall = np.divide(
        np.diag(confusion), confusion.sum(axis=1),
        out=np.zeros(number_of_classes, dtype=float),
        where=confusion.sum(axis=1) != 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(number_of_classes, dtype=float),
        where=precision + recall != 0,
    )
    return precision, recall, f1


def save_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output: Path,
    title: str,
) -> None:
    """Save a labeled confusion matrix with true classes as rows."""
    number_of_classes = len(class_names)
    confusion = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(number_of_classes),
        yticks=range(number_of_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    threshold = confusion.max() / 2 if confusion.size else 0
    for row in range(number_of_classes):
        for column in range(number_of_classes):
            axis.text(
                column, row, str(confusion[row, column]),
                ha="center",
                color="white" if confusion[row, column] > threshold else "black",
            )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def save_metrics_csv(
    history: list[dict[str, float | int]], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def print_class_counts(
    samples: list[tuple[Path, int, str]], class_names: list[str], split: str
) -> None:
    counts = Counter(label for _, label, _ in samples)
    details = ", ".join(f"{name}={counts[index]}" for index, name in enumerate(class_names))
    print(f"{split} samples: {len(samples)} ({details})")


def overfit_sanity_check(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, int]],
    device: torch.device,
    steps: int = 250,
) -> tuple[float, float]:
    """Verify gradients and label alignment on a tiny deterministic slice."""
    count = min(16, len(dataset))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(count)), batch_size=count, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        images, labels = next(iter(loader))
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()
    return run_epoch(model, loader, loss_function, device, None)


def save_batch_preview(
    images: torch.Tensor, labels: torch.Tensor, class_names: list[str], output: Path
) -> None:
    """Save a transformed batch with the real dataset labels."""
    columns = min(4, images.size(0))
    figure, axes = plt.subplots(1, columns, figsize=(3 * columns, 3))
    axes = np.atleast_1d(axes)
    for axis, image, label in zip(axes, images[:columns], labels[:columns]):
        display_image = (image.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        axis.imshow(display_image.numpy())
        axis.set_title(f"dataset label:\n{class_names[label.item()]}", fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)


@torch.no_grad()
def save_prediction_preview(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    class_names: list[str],
    device: torch.device,
    output: Path,
) -> None:
    """Save model predictions separately from the data-pipeline preview."""
    model.eval()
    images, labels = next(iter(loader))
    predictions = model(images.to(device)).argmax(dim=1).cpu()
    columns = min(4, images.size(0))
    figure, axes = plt.subplots(1, columns, figsize=(3 * columns, 3))
    axes = np.atleast_1d(axes)
    for axis, image, label, prediction in zip(
        axes, images[:columns], labels[:columns], predictions[:columns]
    ):
        display_image = (image.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        axis.imshow(display_image.numpy())
        axis.set_title(
            f"true: {class_names[label.item()]}\n"
            f"predicted: {class_names[prediction.item()]}",
            fontsize=9,
        )
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--preview", type=Path, default=Path("outputs/batch_preview.png"))
    parser.add_argument(
        "--prediction-preview",
        type=Path,
        default=Path("outputs/prediction_preview.png"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, validation_samples, test_samples, class_names = collect_samples(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_class_counts(train_samples, class_names, "train")
    print_class_counts(validation_samples, class_names, "validation")
    if test_samples:
        print_class_counts(test_samples, class_names, "test")
    train_dataset = BananaDataset(train_samples, args.image_size, training=True)
    validation_dataset = BananaDataset(validation_samples, args.image_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    test_loader = DataLoader(
        BananaDataset(test_samples, args.image_size),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    ) if test_samples else None
    preview_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    batch_images, batch_labels = next(iter(preview_loader))
    print(f"device={device}; classes={class_names}")
    print(f"batch images={tuple(batch_images.shape)}; labels={tuple(batch_labels.shape)}")
    save_batch_preview(batch_images, batch_labels, class_names, args.preview)
    print(f"saved transformed batch preview to {args.preview}")

    sanity_model = BaselineCNN(len(class_names)).to(device)
    sanity_loss, sanity_accuracy = overfit_sanity_check(sanity_model, train_dataset, device)
    print(f"overfit check: loss={sanity_loss:.4f}, accuracy={sanity_accuracy:.1%}")

    model = BaselineCNN(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy, _, _ = evaluate(
            model, validation_loader, loss_function, device, len(class_names)
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        print(
            f"epoch {epoch:02d}: train loss={train_loss:.4f} acc={train_accuracy:.1%}; "
            f"validation loss={validation_loss:.4f} acc={validation_accuracy:.1%}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "epoch": epoch,
                "validation_loss": validation_loss,
            }, best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break
    save_metrics_csv(history, args.output_dir / "metrics.csv")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"best checkpoint: epoch {best_epoch}, validation loss={best_validation_loss:.4f}")
    validation_loss, validation_accuracy, validation_predictions, validation_labels = evaluate(
        model, validation_loader, loss_function, device, len(class_names)
    )
    precision, recall, f1 = class_metrics(
        validation_predictions, validation_labels, len(class_names)
    )
    for index, name in enumerate(class_names):
        print(
            f"validation {name}: precision={precision[index]:.1%}, "
            f"recall={recall[index]:.1%}, f1={f1[index]:.1%}"
        )
    save_confusion_matrix(
        validation_predictions, validation_labels, class_names,
        args.output_dir / "validation_confusion_matrix.png",
        "Validation confusion matrix",
    )
    if test_loader is not None:
        test_loss, test_accuracy, test_predictions, test_labels = evaluate(
            model, test_loader, loss_function, device, len(class_names)
        )
        print(f"test loss={test_loss:.4f} acc={test_accuracy:.1%}")
        save_confusion_matrix(
            test_predictions, test_labels, class_names,
            args.output_dir / "test_confusion_matrix.png",
            "Test confusion matrix",
        )
    save_prediction_preview(
        model, validation_loader, class_names, device, args.prediction_preview
    )
    print(f"saved model prediction preview to {args.prediction_preview}")


if __name__ == "__main__":
    main()
