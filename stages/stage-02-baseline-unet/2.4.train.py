"""
Stage 2 — Baseline U-Net training.

Experimental protocol:

    37 train.parquet samples
        |
        +---- training subset
        |
        +---- validation subset
                 |
                 v
          choose best checkpoint
                 |
                 v
    14 test.parquet samples
                 |
                 v
        FINAL evaluation

The test set is NOT used for model selection.
"""

import argparse
import os
import random
import importlib.util

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# Load numbered Python files
# ============================================================

def load_module(module_name, filename):

    path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        filename,
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


unet_module = load_module(
    "unet_baseline",
    "2.1.unetbase.py",
)

loss_module = load_module(
    "losses_metrics",
    "2.2.loss metrics.py",
)

dataset_module = load_module(
    "dataset",
    "2.3.dataset.py",
)


UNetBaseline = unet_module.UNetBaseline

BCEDiceLoss = loss_module.BCEDiceLoss
dice_score = loss_module.dice_score
iou_score = loss_module.iou_score

MoNuSegDataset = dataset_module.MoNuSegDataset


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for imgs, masks in loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = loss_fn(
            outputs["nucleus"],
            masks,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * imgs.size(0)
        )

    return (
        total_loss /
        len(loader.dataset)
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device,
):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    n = 0

    for imgs, masks in loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)

        loss = loss_fn(
            outputs["nucleus"],
            masks,
        )

        batch_size = imgs.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_dice += (
            dice_score(
                outputs["nucleus"],
                masks,
            )
            * batch_size
        )

        total_iou += (
            iou_score(
                outputs["nucleus"],
                masks,
            )
            * batch_size
        )

        n += batch_size

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_parquet",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--test_parquet",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--targets_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--base_ch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints_baseline",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    set_seed(args.seed)

    os.makedirs(
        args.ckpt_dir,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 60)
    print("Stage 2 — Baseline U-Net")
    print("=" * 60)

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Load Parquet
    # --------------------------------------------------------

    print()
    print("Loading Parquet datasets...")

    train_df = pd.read_parquet(
        args.train_parquet
    )

    test_df = pd.read_parquet(
        args.test_parquet
    )

    print(
        f"Total train samples: "
        f"{len(train_df)}"
    )

    print(
        f"Total test samples : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Create deterministic validation split
    # --------------------------------------------------------

    indices = np.arange(
        len(train_df)
    )

    rng = np.random.RandomState(
        args.seed
    )

    rng.shuffle(indices)

    n_val = max(
        1,
        int(
            len(indices)
            * args.val_fraction
        ),
    )

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_split_df = (
        train_df.iloc[train_indices]
        .reset_index(drop=True)
    )

    val_split_df = (
        train_df.iloc[val_indices]
        .reset_index(drop=True)
    )

    print()
    print(
        f"Training split  : "
        f"{len(train_split_df)}"
    )

    print(
        f"Validation split: "
        f"{len(val_split_df)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds = MoNuSegDataset(
        dataframe=train_split_df,
        targets_dir=args.targets_dir,
        split="train",
        patch_size=args.patch_size,
        augment=True,
        random_crop=True,
    )

    val_ds = MoNuSegDataset(
        dataframe=val_split_df,
        targets_dir=args.targets_dir,
        split="train",
        patch_size=args.patch_size,
        augment=False,
        random_crop=False,
    )

    test_ds = MoNuSegDataset(
        dataframe=test_df,
        targets_dir=args.targets_dir,
        split="test",
        patch_size=args.patch_size,
        augment=False,
        random_crop=False,
    )

    print()
    print(
        f"Train dataset: "
        f"{len(train_ds)}"
    )

    print(
        f"Val dataset  : "
        f"{len(val_ds)}"
    )

    print(
        f"Test dataset : "
        f"{len(test_ds)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNetBaseline(
        in_channels=3,
        base_ch=args.base_ch,
    ).to(device)

    print()
    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = BCEDiceLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_dice = -1.0

    checkpoint_path = os.path.join(
        args.ckpt_dir,
        "best_baseline.pt",
    )

    print()
    print("=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        val_loss, val_dice, val_iou = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step(val_loss)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_dice {val_dice:.4f} | "
            f"val_iou {val_iou:.4f} | "
            f"lr {current_lr:.2e}"
        )

        # Best model selected ONLY using validation Dice
        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"  -> new best "
                f"(val Dice={best_val_dice:.4f})"
            )

            print(
                f"  -> saved: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Loading best validation checkpoint")
    print("=" * 60)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_loss, test_dice, test_iou = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )

    print()
    print(
        f"Final Test Loss : {test_loss:.4f}"
    )

    print(
        f"Final Test Dice : {test_dice:.4f}"
    )

    print(
        f"Final Test IoU  : {test_iou:.4f}"
    )

    print()
    print("=" * 60)
    print("Stage 2 Baseline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()