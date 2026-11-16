"""
Stage 2 — Dataset loader for the MoNuSeg Parquet dataset.

Actual dataset structure:

    train.parquet
    test.parquet

Each parquet row contains:
    patient   : patient/sample ID
    image     : RGB image stored as bytes
    instances : individual nucleus instance masks
    tissue    : tissue type

Stage 1 generated targets:

    MoNuSeg_targets/
        train/
            binary_masks/
            boundary_masks/
            distance_maps/
            instance_masks/
        test/
            binary_masks/
            boundary_masks/
            distance_maps/
            instance_masks/

For Stage 2 we use:

    Parquet image
          +
    Stage 1 binary mask
          ↓
    image + nucleus mask

The dataset keeps the original train/test split. No random split is
performed here because the dataset already provides an explicit split.
"""

import os

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class MoNuSegDataset(Dataset):
    """
    MoNuSeg dataset using images stored inside Parquet.

    Training:
        random_crop=True
        augment=True

    Validation/Test:
        random_crop=False
        augment=False

    This allows training on 256x256 patches while evaluating on
    the complete 1000x1000 image.
    """

    def __init__(
        self,
        dataframe,
        targets_dir,
        split="train",
        patch_size=256,
        augment=False,
        random_crop=True,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.targets_dir = targets_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.random_crop = random_crop

        self.binary_mask_dir = os.path.join(
            self.targets_dir,
            self.split,
            "binary_masks",
        )

    def __len__(self):
        return len(self.df)

    def _load_image(self, row):
        image_bytes = row["image"]["bytes"]

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image = np.array(image).astype(
            np.float32
        ) / 255.0

        return image

    def _load_binary_mask(self, patient):
        path = os.path.join(
            self.binary_mask_dir,
            patient + ".npy",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Binary mask not found:\n{path}"
            )

        mask = np.load(path).astype(
            np.float32
        )

        mask = (mask > 0).astype(
            np.float32
        )

        return mask

    def _random_crop(self, img, mask):
        h, w = mask.shape
        ps = self.patch_size

        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)

            img = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )

            mask = np.pad(
                mask,
                ((0, pad_h), (0, pad_w)),
                mode="reflect",
            )

            h, w = mask.shape

        top = np.random.randint(
            0,
            h - ps + 1,
        )

        left = np.random.randint(
            0,
            w - ps + 1,
        )

        img = img[
            top:top + ps,
            left:left + ps,
        ]

        mask = mask[
            top:top + ps,
            left:left + ps,
        ]

        return img, mask

    def _augment(self, img, mask):

        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if np.random.rand() < 0.5:
            img = img[::-1, :].copy()
            mask = mask[::-1, :].copy()

        k = np.random.randint(0, 4)

        if k:
            img = np.rot90(
                img,
                k,
                axes=(0, 1),
            ).copy()

            mask = np.rot90(
                mask,
                k,
                axes=(0, 1),
            ).copy()

        return img, mask

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        patient = str(row["patient"])

        img = self._load_image(row)

        mask = self._load_binary_mask(
            patient
        )

        if img.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask mismatch for {patient}: "
                f"image={img.shape}, "
                f"mask={mask.shape}"
            )

        # Training → random 256x256 crop
        if self.random_crop:
            img, mask = self._random_crop(
                img,
                mask,
            )

        # Training augmentation only
        if self.augment:
            img, mask = self._augment(
                img,
                mask,
            )

        img_t = torch.from_numpy(
            img.transpose(2, 0, 1)
        ).float()

        mask_t = torch.from_numpy(
            mask
        ).float().unsqueeze(0)

        return img_t, mask_t
    
if __name__ == "__main__":
    """
    Real-data smoke test.

    Update these two paths before running:

        PARQUET_PATH
        TARGETS_DIR
    """

    PARQUET_PATH = r"C:\Users\Dell\OneDrive\Documents\SEGMENTATION\train-00000-of-00001.parquet"
    TARGETS_DIR = r"C:\Users\Dell\Desktop\MoNuSeg_targets"

    print("=" * 60)
    print("Stage 2 Dataset Smoke Test")
    print("=" * 60)

    # Load real Parquet data
    df = load_parquet_split(PARQUET_PATH)

    print(f"Samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Create dataset
    dataset = MoNuSegDataset(
        dataframe=df,
        targets_dir=TARGETS_DIR,
        split="train",
        patch_size=256,
        augment=True,
    )

    print(f"Dataset length: {len(dataset)}")

    # Load one real sample
    img_t, mask_t = dataset[0]

    patient = str(df.iloc[0]["patient"])

    print()
    print("Patient:", patient)
    print("Image tensor:", img_t.shape, img_t.dtype)
    print("Image range:", img_t.min().item(), img_t.max().item())
    print("Mask tensor:", mask_t.shape, mask_t.dtype)
    print("Mask unique values:", torch.unique(mask_t))
    print("Positive pixels:", mask_t.sum().item())

    # Shape checks
    assert img_t.shape == (
        3,
        256,
        256,
    ), "Image shape mismatch!"

    assert mask_t.shape == (
        1,
        256,
        256,
    ), "Mask shape mismatch!"

    # Binary mask check
    unique_values = torch.unique(mask_t)

    assert torch.all(
        (unique_values == 0) |
        (unique_values == 1)
    ), "Mask is not binary!"

    print()
    print("OK: real MoNuSeg Parquet dataset smoke test passed.")

