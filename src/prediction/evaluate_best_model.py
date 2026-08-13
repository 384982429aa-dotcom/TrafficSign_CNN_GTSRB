# -*- coding: utf-8 -*-
"""Evaluate the saved best checkpoint on validation and official test sets.

Run from the project root:
    python src\prediction\evaluate_best_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_model import build_cnn_model  # noqa: E402
from src.training.train import (  # noqa: E402
    GTSRBDataset,
    apply_label_mapping,
    load_csv_dataframe,
    save_report_and_confusion_matrix,
    set_seed,
)


DEFAULT_DATA_DIR = r"E:\ML\GTSRB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate resnet18_fpn_best.pth on GTSRB validation and test sets."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "resnet18_fpn_best.pth"),
    )
    parser.add_argument("--outputs-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Print evaluation progress every N batches; use 0 to disable.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Best model not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    required = {"model_state_dict", "num_classes", "class_to_index", "index_to_class"}
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing fields: {sorted(missing)}")
    return checkpoint


@torch.no_grad()
def evaluate_split(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    num_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_true.append(labels.cpu().numpy())
        all_pred.append(predictions.cpu().numpy())

        should_log = log_interval > 0 and (
            batch_index == 1
            or batch_index % log_interval == 0
            or batch_index == num_batches
        )
        if should_log:
            print(f"{name}: batch {batch_index}/{num_batches}", flush=True)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return total_loss / total, accuracy_score(y_true, y_pred), y_true, y_pred


def metric_summary(loss: float, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def make_loader(dataset: GTSRBDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def print_summary(name: str, metrics: dict[str, float], sample_count: int) -> None:
    print(
        f"{name}: samples={sample_count}, loss={metrics['loss']:.6f}, "
        f"accuracy={metrics['accuracy']:.4%}, macro_f1={metrics['macro_f1']:.4f}, "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    outputs_dir = Path(args.outputs_dir)
    reports_dir = outputs_dir / "reports"
    figures_dir = outputs_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(model_path, device)
    num_classes = int(checkpoint["num_classes"])
    img_size = int(checkpoint.get("img_size", 64))
    class_to_index = {int(k): int(v) for k, v in checkpoint["class_to_index"].items()}
    index_to_class = {int(k): int(v) for k, v in checkpoint["index_to_class"].items()}

    model = build_cnn_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Checkpoint best val_acc: {float(checkpoint.get('val_acc', 0.0)):.4%}")
    print(f"Image size: {img_size}x{img_size}")

    train_df = load_csv_dataframe(data_dir, "Train.csv")
    train_df = apply_label_mapping(train_df, class_to_index)
    train_split, val_split = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["label_index"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_split["label_index"].to_numpy(),
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    val_dataset = GTSRBDataset(val_split, img_size=img_size, training=False)
    val_loader = make_loader(val_dataset, args)
    val_loss, _, val_true, val_pred = evaluate_split(
        "Validation",
        model,
        val_loader,
        criterion,
        device,
        args.log_interval,
    )
    val_metrics = metric_summary(val_loss, val_true, val_pred)
    save_report_and_confusion_matrix(
        val_true,
        val_pred,
        index_to_class,
        reports_dir / "best_validation_classification_report.txt",
        figures_dir / "best_validation_confusion_matrix.png",
        "Best Model - Validation Confusion Matrix",
    )

    test_df = load_csv_dataframe(data_dir, "Test.csv")
    test_df = apply_label_mapping(test_df, class_to_index)
    test_dataset = GTSRBDataset(test_df, img_size=img_size, training=False)
    test_loader = make_loader(test_dataset, args)
    test_loss, _, test_true, test_pred = evaluate_split(
        "Official test",
        model,
        test_loader,
        criterion,
        device,
        args.log_interval,
    )
    test_metrics = metric_summary(test_loss, test_true, test_pred)
    save_report_and_confusion_matrix(
        test_true,
        test_pred,
        index_to_class,
        reports_dir / "best_test_classification_report.txt",
        figures_dir / "best_test_confusion_matrix.png",
        "Best Model - Official Test Confusion Matrix",
    )

    summary = {
        "model_path": str(model_path),
        "checkpoint_best_val_acc": float(checkpoint.get("val_acc", 0.0)),
        "img_size": img_size,
        "seed": args.seed,
        "val_size": args.val_size,
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "validation": val_metrics,
        "official_test": test_metrics,
    }
    summary_path = reports_dir / "best_model_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nBest model evaluation completed")
    print_summary("Validation", val_metrics, len(val_dataset))
    print_summary("Official test", test_metrics, len(test_dataset))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
