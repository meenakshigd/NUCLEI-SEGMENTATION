# Stage 2 — Baseline U-Net

**Purpose:** Train and evaluate the single-head baseline U-Net for nucleus-mask prediction.

**Available source files:**
- 2.1.unetbase.py — baseline U-Net architecture.
- 2.2.loss metrics.py — BCE-Dice loss and pixel-level metrics.
- 2.3.dataset.py — MoNuSeg dataset loader.
- 2.4.train.py — training and evaluation script.

The training script expects local Parquet files and a Stage 1 target directory. Configure these paths before running. The dataset smoke-test section also contains example machine-specific paths.
