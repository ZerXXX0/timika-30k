"""
Unified Training Script for Multi-Model CXR Segmentation.
Models: U-Net++, FPN, SegFormer (MiT)
Scenarios: CLAHE, Bone-Suppressed, Bone-Suppression + CLAHE
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.config import Hyperparameters, SCENARIOS, DISEASE_CLASSES
from experiments.data.dataset import TimikaDiseaseDataset
from experiments.losses.compound_loss import (
    MaskedMultiLabelBCEDiceLoss,
    MaskedFocalTverskyLoss,
)
from experiments.models.factory import build_model
from experiments.metrics import SegmentationMetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train CXR Segmentation Models across Scenarios")
    parser.add_argument("--model", type=str, default="unetplusplus", choices=["unetplusplus", "fpn", "segformer"])
    parser.add_argument("--scenario", type=str, default="bone_suppression_clahe", choices=["clahe", "bone_suppressed", "bone_suppression_clahe"])
    parser.add_argument("--encoder", type=str, default=None, help="Backbone encoder (e.g. convnext_small, resnet34, mit_b3)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="masked_bce_dice", choices=["masked_bce_dice", "masked_focal_tversky"])
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Subsample for smoke tests/quick trials")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    metric_tracker = SegmentationMetricTracker()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        annotated_mask = batch["annotated_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True):
            logits = model(images)
            loss = criterion(logits, targets, annotated_mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        metric_tracker.update(logits.detach(), targets, annotated_mask)

    epoch_loss = total_loss / max(len(loader), 1)
    metrics = metric_tracker.compute()
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metric_tracker = SegmentationMetricTracker()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        annotated_mask = batch["annotated_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True):
            logits = model(images)
            loss = criterion(logits, targets, annotated_mask)

        total_loss += loss.item()
        metric_tracker.update(logits, targets, annotated_mask)

    epoch_loss = total_loss / max(len(loader), 1)
    metrics = metric_tracker.compute()
    metrics["loss"] = epoch_loss
    return metrics


def main():
    args = parse_args()

    # Assign default encoder based on architecture
    if args.encoder is None:
        if args.model == "segformer":
            args.encoder = "mit_b3"
        elif args.model == "unetplusplus":
            args.encoder = "resnet34"
        else:
            args.encoder = "resnet34"

    experiment_name = f"{args.model}_{args.encoder}_{args.scenario}"
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model: {args.model.upper()} | Encoder: {args.encoder} | Scenario: {SCENARIOS[args.scenario]['name']}")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs} | Loss: {args.loss}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Save directories
    checkpoints_dir = os.path.join("experiments", "checkpoints", experiment_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Datasets and Loaders
    train_dataset = TimikaDiseaseDataset(
        scenario=args.scenario,
        split="train",
        max_samples=args.max_train_samples,
    )
    val_dataset = TimikaDiseaseDataset(
        scenario=args.scenario,
        split="val",
        max_samples=args.max_train_samples // 4 if args.max_train_samples else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2. Build Model
    model = build_model(
        model_name=args.model,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=1,
        num_classes=len(DISEASE_CLASSES),
    ).to(args.device)

    # 3. Criterion, Optimizer, Scheduler, Scaler
    if args.loss == "masked_bce_dice":
        criterion = MaskedMultiLabelBCEDiceLoss(
            bce_weight=args.bce_weight, dice_weight=args.dice_weight
        )
    else:
        criterion = MaskedFocalTverskyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler(enabled=True)

    # 4. Training Loop with Early Stopping
    best_macro_dice = 0.0
    patience_counter = 0
    history = []
    start_epoch = 1

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n=> Loading checkpoint to resume: {args.resume}")
            ckpt = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            saved_epoch = ckpt.get("epoch", 0)
            start_epoch = saved_epoch + 1
            best_macro_dice = ckpt.get("best_macro_dice", 0.0)

            # Advance learning rate scheduler to match resumed epoch
            for _ in range(1, start_epoch):
                scheduler.step()

            # Load existing history CSV if present
            prev_history_path = os.path.join(os.path.dirname(args.resume), "training_history.csv")
            if os.path.exists(prev_history_path):
                try:
                    df_prev = pd.read_csv(prev_history_path)
                    df_prev = df_prev[df_prev["epoch"] <= saved_epoch]
                    history = df_prev.to_dict("records")
                    print(f"=> Loaded {len(history)} previous epoch records from {prev_history_path}")
                except Exception as e:
                    print(f"=> Note: Could not read prior history file: {e}")

            print(f"=> Resuming from epoch {saved_epoch} (Best Val Dice: {best_macro_dice:.4f}). Starting epoch {start_epoch}.\n")
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {args.resume}")

    history_path = os.path.join(checkpoints_dir, "training_history.csv")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, args.device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, args.device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_curr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro Dice: {val_metrics['macro_dice']:.4f} | LR: {lr_curr:.2e}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_macro_dice": train_metrics["macro_dice"],
            "val_macro_dice": val_metrics["macro_dice"],
            "lr": lr_curr,
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in ["loss", "macro_dice"]},
        }
        history.append(record)

        # Incrementally persist training history to avoid loss on unexpected interruption
        pd.DataFrame(history).to_csv(history_path, index=False)

        # Checkpoint if best validation Macro Dice
        if val_metrics["macro_dice"] > best_macro_dice:
            best_macro_dice = val_metrics["macro_dice"]
            patience_counter = 0
            best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_dice": best_macro_dice,
                    "config": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint: {best_ckpt_path} (Dice: {best_macro_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"\n[Early Stopping Triggered] No improvement for {args.early_stopping} consecutive epochs.")
                break

    print(f"\nTraining completed. Full history saved to {history_path}")


if __name__ == "__main__":
    main()

