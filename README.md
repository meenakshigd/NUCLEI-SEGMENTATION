# Nuclei Instance Segmentation

Boundary- and distance-aware multi-task U-Net for nuclei instance segmentation in H&E-stained histopathology images.

## Research pipeline
1. Target generation — binary, instance, boundary and distance-transform targets.
2. Baseline U-Net — nucleus-mask prediction.
3. Boundary-aware U-Net — nucleus and boundary prediction.
4. Multi-task U-Net — nucleus, boundary and distance prediction.
5. Inference and evaluation — marker-controlled watershed, Dice, IoU and AJI.

## Results
Mean performance on the 14 held-out MoNuSeg test images:

| Model | Dice | IoU |
|---|---:|---:|
| Baseline U-Net | 0.5695 | 0.4004 |
| Boundary-aware U-Net | 0.6244 | 0.4581 |
| Multi-task U-Net | **0.6735** | **0.5123** |

Stage 4 watershed reconstruction achieved mean AJI **0.4045** over 6,697 ground-truth nuclei.

## Repository guide
- [Implementation stages](stages/)
- [Methodology](docs/methodology.md)
- [Results](docs/results.md)
- [Research paper](paper/README.md)
- [Project reports](reports/README.md)

## Dataset
The study uses MoNuSeg 2018. Dataset files are not included; obtain them from the official source and follow their terms of use.

## Running the code
The current repository snapshot contains exported HTML artifacts for Stages 1, 4 and 5, and Python source files for Stage 2. Stage 3 source files were not present in the inspected snapshot. Some scripts require local dataset paths to be configured before execution. See the README in each stage folder.

## Publication
**Boundary and Distance-Aware Multi-Task U-Net for Nuclei Instance Segmentation in Histopathological Images**

Authors: Meenakshi G, Kabel B, and Dayalin Leena G  
Conference: ICCWN 2026  
Status: Accepted; publication forthcoming.

The DOI and official proceedings link will be added when issued.

## Scope
Morphological feature analysis, spatial graph construction and graph-neural-network-based cancer classification are future directions, not implemented components of this study.

## License
Copyright © 2026 Meenakshi Ganapathy. All rights reserved. See [LICENSE](LICENSE).
