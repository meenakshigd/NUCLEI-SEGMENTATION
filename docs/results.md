# Results

Values below are transcribed from the research paper.

## Pixel-level evaluation
Mean ± standard deviation over the 14 held-out test images:

| Model | Dice | IoU |
|---|---:|---:|
| Stage 2 — Baseline U-Net | 0.5695 ± 0.0578 | 0.4004 ± 0.0552 |
| Stage 3 — Boundary-aware U-Net | 0.6244 ± 0.0755 | 0.4581 ± 0.0763 |
| Stage 4 — Multi-task U-Net | **0.6735 ± 0.0760** | **0.5123 ± 0.0807** |

## Instance-level evaluation
Stage 4 watershed reconstruction achieved mean AJI **0.4045 ± 0.0486** across 14 test images containing 6,697 ground-truth nuclei. The paper reports 5,442 predicted instances.

## Figure assets
The accepted paper contains seven figures. Original extracted figure images should be placed in results/figures with their original numbering and captions. Do not substitute recreated or illustrative graphics.

## Limitations
The study uses a relatively small dataset and test set, 256 × 256 inputs, a single training run and data split. Watershed performance is sensitive to marker-extraction settings, and generalization to additional datasets and staining conditions requires further validation.
