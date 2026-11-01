"""Pure-PyTorch baseline for banana ripeness classification.

The default five-class mode splits the original ``ripe`` directory into two
deterministic pseudo-classes using mean HSV hue. This is a baseline for
experimentation, not a replacement for verified labels.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("unripe", "partially_ripe", "ripe", "overripe", "rotten")


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


def hue_score(path: Path) -> float:
    """Return mean HSV hue, used only to make the pseudo-label split."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    difference = maximum - minimum
    hue = np.zeros_like(maximum)
    red = (maximum == rgb[:, :, 0]) & (difference > 0)
    green = (maximum == rgb[:, :, 1]) & (difference > 0)
    blue = (maximum == rgb[:, :, 2]) & (difference > 0)
    hue[red] = ((rgb[:, :, 1][red] - rgb[:, :, 2][red]) / difference[red]) % 6
    hue[green] = (rgb[:, :, 2][green] - rgb[:, :, 0][green]) / difference[green] + 2
    hue[blue] = (rgb[:, :, 0][blue] - rgb[:, :, 1][blue]) / difference[blue] + 4
    return float((hue / 6.0).mean())


def collect_samples(root: Path, five_classes: bool) -> tuple[list[tuple[Path, int]], list[str]]:
    """Discover split/class folders and return image paths with integer labels."""
    class_dirs = [path for path in root.rglob("*") if path.is_dir() and any(
        child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir()
        if child.is_file()
    )]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories with images found below {root}")

    names = sorted({path.name.lower() for path in class_dirs})
    required = {"unripe", "ripe", "overripe", "rotten"}
    if not required.issubset(names):
        raise ValueError(
            f"Expected class directories containing {sorted(required)}, found {names}"
        )

    class_names = list(CLASS_NAMES if five_classes else ("unripe", "ripe", "overripe", "rotten"))
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(class_dirs):
        source_name = class_dir.name.lower()
        files = sorted(path for path in class_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if source_name == "ripe" and five_classes:
            scored = sorted((hue_score(path), path) for path in files)
            midpoint = len(scored) // 2
            samples.extend((path, 1 if index < midpoint else 2)
                           for index, (_, path) in enumerate(scored))
        else:
            label = class_names.index(source_name)
            samples.extend((path, label) for path in files)
    if not samples:
        raise ValueError(f"No images found below {root}")
    return samples, class_names


class BananaDataset(Dataset[tuple[torch.Tensor, int]]):
    """Custom Dataset demonstrating explicit __len__ and __getitem__."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int = 128,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = image_to_tensor(image, self.image_size, self.training)
        return tensor, label


class BaselineCNN(nn.Module):
    """Small CNN whose tensor transitions are easy to explain at the defense."""

    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # B,3,128,128 -> B,16,128,128
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> B,16,64,64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> B,32,64,64
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> B,32,32,32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> B,64,32,32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> B,64,1,1
        )
        self.classifier = nn.Linear(64, number_of_classes)  # -> B,5

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
    """Save one post-transformation batch so tensor contents can be inspected."""
    columns = min(4, images.size(0))
    figure, axes = plt.subplots(1, columns, figsize=(3 * columns, 3))
    axes = np.atleast_1d(axes)
    for axis, image, label in zip(axes, images[:columns], labels[:columns]):
        display_image = (image.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        axis.imshow(display_image.numpy())
        axis.set_title(class_names[label.item()])
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--four-classes", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preview", type=Path, default=Path("outputs/batch_preview.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, class_names = collect_samples(args.data, five_classes=not args.four_classes)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples, validation_samples = samples[:split], samples[split:]
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
    batch_images, batch_labels = next(iter(train_loader))
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
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function, device, None
        )
        print(
            f"epoch {epoch:02d}: train loss={train_loss:.4f} acc={train_accuracy:.1%}; "
            f"validation loss={validation_loss:.4f} acc={validation_accuracy:.1%}"
        )


if __name__ == "__main__":
    main()
