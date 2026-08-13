# -*- coding: utf-8 -*-
"""
Train PyTorch Small-Image ResNet18-FPN on GTSRB.

Dataset root:
    E:\\ML\\GTSRB

Expected files:
    E:\\ML\\GTSRB\\data\\Train.csv
    E:\\ML\\GTSRB\\data\\Test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_model import build_cnn_model  # noqa: E402


DEFAULT_DATA_DIR = r"E:\ML\GTSRB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch ResNet18-FPN for GTSRB.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100, help="Print progress every N training batches.")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--no-official-test", action="store_true")
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--outputs-dir", default=str(PROJECT_ROOT / "outputs"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_csv(data_dir: Path, csv_name: str) -> Path:
    candidates = [data_dir / "data" / csv_name, data_dir / csv_name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {csv_name}. Tried: {candidates}")


def load_csv_dataframe(data_dir: Path, csv_name: str) -> pd.DataFrame:
    csv_path = find_csv(data_dir, csv_name)
    df = pd.read_csv(csv_path)

    required = {"Path", "ClassId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["image_path"] = df["Path"].apply(lambda p: str(data_dir / str(p).replace("/", os.sep)))
    df["ClassId"] = df["ClassId"].astype(int)

    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count:
        print(f"Warning: skipped {missing_count} missing images from {csv_path.name}")
        df = df[exists_mask].reset_index(drop=True)

    return df


def create_label_mapping(labels: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    class_ids = sorted(int(x) for x in np.unique(labels))
    class_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    index_to_class = {idx: class_id for class_id, idx in class_to_index.items()}
    return class_to_index, index_to_class


def apply_label_mapping(df: pd.DataFrame, class_to_index: dict[int, int]) -> pd.DataFrame:
    df = df.copy()
    unknown = sorted(set(df["ClassId"].astype(int)) - set(class_to_index))
    if unknown:
        raise ValueError(f"Unknown class ids not found in training set: {unknown}")
    df["label_index"] = df["ClassId"].map(class_to_index).astype(int)
    return df


class GTSRBDataset(Dataset):
    """GTSRB dataset loaded from CSV paths."""

    def __init__(self, df: pd.DataFrame, img_size: int, training: bool):
        self.paths = df["image_path"].tolist()
        self.labels = df["label_index"].astype(int).tolist()
        self.img_size = img_size
        self.training = training

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = cv2.imread(self.paths[index])
        if image is None:
            raise FileNotFoundError(self.paths[index])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.training:
            image = self._augment(image)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image)
        label = self.labels[index]
        return tensor, label

    def _augment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        if random.random() < 0.8:
            angle = random.uniform(-10, 10)
            scale = random.uniform(0.9, 1.1)
            tx = random.uniform(-0.08, 0.08) * w
            ty = random.uniform(-0.08, 0.08) * h
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image


def make_dirs(models_dir: Path, outputs_dir: Path) -> tuple[Path, Path, Path]:
    figures_dir = outputs_dir / "figures"
    reports_dir = outputs_dir / "reports"
    logs_dir = outputs_dir / "logs"
    for path in [models_dir, figures_dir, reports_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return figures_dir, reports_dir, logs_dir


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == num_batches):
            avg_loss = total_loss / total
            avg_acc = correct / total
            print(
                f"Epoch {epoch:03d} [{batch_idx:04d}/{num_batches}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_true = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        all_pred.append(pred.cpu().numpy())
        all_true.append(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )


def save_history(history: list[dict[str, float]], save_path: Path) -> None:
    pd.DataFrame(history).to_csv(save_path, index=False, encoding="utf-8-sig")


def plot_training_curves(history: list[dict[str, float]], save_path: Path) -> None:
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train_acc")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_report_and_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index_to_class: dict[int, int],
    report_path: Path,
    cm_path: Path,
    title: str,
) -> None:
    labels = list(range(len(index_to_class)))
    target_names = [str(index_to_class[i]) for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", square=True, cbar=True)
    plt.xlabel("Predicted label index")
    plt.ylabel("True label index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir, reports_dir, logs_dir = make_dirs(models_dir, outputs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {data_dir}")

    train_df = load_csv_dataframe(data_dir, "Train.csv")
    class_to_index, index_to_class = create_label_mapping(train_df["ClassId"].to_numpy())
    train_df = apply_label_mapping(train_df, class_to_index)
    num_classes = len(class_to_index)

    print(f"Train samples: {len(train_df)}")
    print(f"Detected classes: {num_classes}")

    mapping_payload = {
        "class_to_index": {str(k): int(v) for k, v in class_to_index.items()},
        "index_to_class": {str(k): int(v) for k, v in index_to_class.items()},
        "num_classes": num_classes,
        "img_size": args.img_size,
    }
    (models_dir / "label_mapping.json").write_text(
        json.dumps(mapping_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_split, val_split = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["label_index"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    train_dataset = GTSRBDataset(train_split, img_size=args.img_size, training=True)
    val_dataset = GTSRBDataset(val_split, img_size=args.img_size, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_cnn_model(num_classes=num_classes).to(device)

    if args.no_class_weight:
        class_weights = None
    else:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=train_split["label_index"].to_numpy(),
        )
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_acc = 0.0
    history = []
    best_model_path = models_dir / "resnet18_fpn_best.pth"
    final_model_path = models_dir / "resnet18_fpn_final.pth"

    print(f"Start training: {len(train_loader)} train batches/epoch, {len(val_loader)} validation batches/epoch", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "class_to_index": class_to_index,
                    "index_to_class": index_to_class,
                    "val_acc": best_acc,
                },
                best_model_path,
            )
            print(f"Saved best model: {best_model_path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "img_size": args.img_size,
            "class_to_index": class_to_index,
            "index_to_class": index_to_class,
            "val_acc": best_acc,
        },
        final_model_path,
    )

    save_history(history, logs_dir / "training_log.csv")
    plot_training_curves(history, figures_dir / "training_curves.png")

    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    save_report_and_confusion_matrix(
        y_true,
        y_pred,
        index_to_class,
        reports_dir / "validation_classification_report.txt",
        figures_dir / "validation_confusion_matrix.png",
        "Validation Confusion Matrix",
    )

    if not args.no_official_test:
        test_df = load_csv_dataframe(data_dir, "Test.csv")
        test_df = apply_label_mapping(test_df, class_to_index)
        test_dataset = GTSRBDataset(test_df, img_size=args.img_size, training=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        _, _, test_true, test_pred = evaluate(model, test_loader, criterion, device)
        save_report_and_confusion_matrix(
            test_true,
            test_pred,
            index_to_class,
            reports_dir / "test_classification_report.txt",
            figures_dir / "test_confusion_matrix.png",
            "Official Test Confusion Matrix",
        )

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")


if __name__ == "__main__":
    main()
