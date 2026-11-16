# Methodology

## Objective
Separate individual nuclei in H&E-stained histopathology images by combining nucleus prediction with boundary and distance cues.

## Dataset
MoNuSeg 2018: 37 training images (32 training and 5 validation) and 14 held-out test images. The test set contains 6,697 ground-truth nuclei. Dataset files are not redistributed here.

## Five-stage workflow
1. Target generation: derive instance, binary, boundary and per-nucleus normalized distance-transform targets.
2. Baseline U-Net: train a single-head model for nucleus-mask prediction.
3. Boundary-aware U-Net: add a boundary prediction task.
4. Multi-task U-Net: jointly predict nucleus, boundary and distance maps.
5. Inference and evaluation: use marker-controlled watershed for instance reconstruction; evaluate Dice, IoU and AJI.

## Model and training
The paper reports a base width of 32, four down-sampling blocks, 256 × 256 training patches, 50 epochs and 7,766,051 trainable parameters for the full multi-task model. Training was performed on CPU.

## Reproducibility
The inspected repository snapshot contains exported HTML artifacts for Stages 1, 4 and 5, plus Stage 2 Python files. Stage 3 source files were not present. Some Stage 2 paths are machine-specific and require local configuration.
