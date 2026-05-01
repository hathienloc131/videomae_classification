"""
Training script for VideoMAE task-completion classifier.

Usage:
    python train.py
    python train.py --dataset_root /Volumes/LocHT/lerobot_vrh31 --epochs 30 --batch_size 8
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import LABEL_NAMES, NUM_CLASSES, TrainConfig
from dataset import LeRobotClipDataset
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Train VideoMAE task-completion classifier")
    p.add_argument("--dataset_root", default=cfg.dataset_root)
    p.add_argument("--camera", default=cfg.camera)
    p.add_argument("--checkpoint_dir", default=cfg.checkpoint_dir)
    p.add_argument("--model_name", default=cfg.model_name)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--clip_frames", type=int, default=cfg.clip_frames)
    p.add_argument("--clip_stride", type=int, default=cfg.clip_stride)
    p.add_argument("--num_frames", type=int, default=cfg.num_frames)
    p.add_argument("--image_size", type=int, default=cfg.image_size)
    p.add_argument("--val_split", type=float, default=cfg.val_split)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Only train the classification head")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name. Omit to disable W&B logging.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (optional)")
    return p.parse_args()


def compute_class_weights(dataset: LeRobotClipDataset) -> torch.Tensor:
    counts = np.bincount([r[3] for r in dataset.records], minlength=NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32)
    return weights / weights.sum() * NUM_CLASSES  # normalise so avg weight ≈ 1


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    total_loss = correct = total = 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        preds = outputs.logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    print(f"Device: {device}, fp16: {use_fp16}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, "logs"))

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Pass empty list → dataset.py auto-scans for sub-folders with meta/info.json
    shared_kwargs = dict(
        dataset_root=args.dataset_root,
        dataset_subdirs=[],
        camera=args.camera,
        clip_frames=args.clip_frames,
        clip_stride=args.clip_stride,
        num_frames=args.num_frames,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    train_ds = LeRobotClipDataset(split="train", **shared_kwargs)
    val_ds = LeRobotClipDataset(split="val", **shared_kwargs)

    label_counts = np.bincount([r[3] for r in train_ds.records], minlength=NUM_CLASSES)
    print(f"Train clips: {len(train_ds)}  {dict(zip(LABEL_NAMES.values(), label_counts))}")
    print(f"Val   clips: {len(val_ds)}")

    # Weighted sampler to handle class imbalance
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model_name, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # ── Loss with class weighting ──────────────────────────────────────────────
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,          # 10% warmup
        anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = running_correct = running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=use_fp16):
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = outputs.logits.argmax(dim=-1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            global_step += 1

            if global_step % 20 == 0:
                train_acc = running_correct / running_total
                train_loss = running_loss / running_total
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.3f}")
                writer.add_scalar("train/loss", train_loss, global_step)
                writer.add_scalar("train/acc", train_acc, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if use_wandb:
                    wandb.log({"train/loss": train_loss, "train/acc": train_acc, "train/lr": lr}, step=global_step)

        val_loss, val_acc = evaluate(model, val_loader, device, scaler if use_fp16 else None)
        print(f"  [val] loss={val_loss:.4f}  acc={val_acc:.4f}")
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        if use_wandb:
            wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch}, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc},
                ckpt_path,
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f}) → {ckpt_path}")

    # Save final checkpoint
    torch.save(
        {"epoch": args.epochs, "model": model.state_dict(), "val_acc": val_acc},
        os.path.join(args.checkpoint_dir, "final_model.pth"),
    )
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    writer.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
