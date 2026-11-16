<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:160b18,35:552044,70:9d426b,100:e6a0ad&height=220&section=header&text=NUCLEI%20SEGMENTATION&fontSize=42&fontColor=fff4f7&fontAlignY=38&desc=Seeing%20the%20Cellular%20Story%20Behind%20Tissue&descAlignY=60&descSize=16&animation=fadeIn&fontFamily=Verdana"/>

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=19&pause=1200&color=E6A0AD&center=true&vCenter=true&width=850&lines=Boundary-Aware+Nuclei+Segmentation;Distance-Guided+Multi-Task+Learning;From+Histopathology+Pixels+to+Instance+Masks;A+Deep+Learning+Study+on+MoNuSeg+2018" alt="Project tagline"/>

<br/>

<img src="https://img.shields.io/badge/DOMAIN-Digital%20Pathology-6D294F?style=for-the-badge&labelColor=21101E"/>
<img src="https://img.shields.io/badge/TASK-Nuclei%20Instance%20Segmentation-BB6688?style=for-the-badge&labelColor=21101E"/>
<img src="https://img.shields.io/badge/MODEL-Multi--Task%20U--Net-DA91A8?style=for-the-badge&labelColor=21101E"/>
<img src="https://img.shields.io/badge/DATASET-MoNuSeg%202018-8C6AA8?style=for-the-badge&labelColor=21101E"/>

<br/><br/>

**A boundary- and distance-aware deep learning pipeline for separating individual nuclei in H&E-stained histopathology images.**

[Research Pipeline](#-the-research-journey) · [Key Results](#-results-at-a-glance) · [Repository Guide](#-explore-the-repository) · [Publication](#-research-publication)

</div>

---

## 🧬 The Cellular Challenge

In histopathology, tissue is more than a collection of pixels. It is a complex landscape of cells, nuclei, boundaries, and spatial relationships.

Accurately identifying each nucleus is challenging when neighbouring nuclei touch, overlap, or have irregular shapes. This project investigates a **multi-task U-Net** that learns complementary nuclear representations — the nucleus region, its boundary, and its distance from the boundary — to support more precise instance segmentation.

<div align="center">

| 🔬 Research focus | 🧫 Learning strategy | 🧠 Reconstruction |
|:---:|:---:|:---:|
| H&E-stained tissue images | Joint mask, boundary & distance prediction | Marker-controlled watershed |

</div>

## ✨ What This Research Explores

- 🟣 **Nucleus segmentation** — predicts the foreground region occupied by nuclei.
- 🩷 **Boundary awareness** — learns separating contours between neighbouring nuclei.
- 🧭 **Distance guidance** — learns a normalized distance-transform representation.
- 🧩 **Instance reconstruction** — uses marker-controlled watershed to separate connected regions.
- 📊 **Quantitative evaluation** — reports pixel-level Dice and IoU, alongside instance-level AJI.

## 🧪 The Research Journey

The study is organized as a progressive five-stage pipeline. Each stage adds a learning capability or evaluation step.

<div align="center">

| Stage | Research component | What happens |
|:---:|---|---|
| **01** 🧫 | Target Generation | Creates instance, binary, boundary and distance-transform targets |
| **02** 🧠 | Baseline U-Net | Learns nucleus-mask prediction |
| **03** 🩷 | Boundary-Aware U-Net | Learns nucleus and boundary prediction |
| **04** 🧭 | Multi-Task U-Net | Jointly predicts nucleus, boundary and distance maps |
| **05** 🧩 | Inference & Evaluation | Reconstructs instances with watershed and evaluates predictions |

</div>

> **The central idea:** instead of asking the network to learn only *where a nucleus is*, the model also learns *where its boundary lies* and *how far pixels are from that boundary*.

## 📊 Results at a Glance

Mean pixel-level performance on the **14 held-out MoNuSeg test images**.

<div align="center">

| Model progression | Dice ↑ | IoU ↑ |
|:---|:---:|:---:|
| Baseline U-Net | 0.5695 | 0.4004 |
| Boundary-Aware U-Net | 0.6244 | 0.4581 |
| **Multi-Task U-Net** | **0.6735** | **0.5123** |

<br/>

<img src="https://img.shields.io/badge/Best%20Mean%20Dice-0.6735-8D3B65?style=for-the-badge&labelColor=21101E"/>
<img src="https://img.shields.io/badge/Best%20Mean%20IoU-0.5123-BB6688?style=for-the-badge&labelColor=21101E"/>
<img src="https://img.shields.io/badge/Mean%20AJI-0.4045-8C6AA8?style=for-the-badge&labelColor=21101E"/>

</div>

**Instance-level result:** Stage 4 marker-controlled watershed reconstruction achieved a mean **Aggregated Jaccard Index (AJI) of 0.4045 ± 0.0486**, evaluated across **6,697 ground-truth nuclei**.

These results describe this study's experimental setup; they should not be interpreted as a direct leaderboard comparison with methods evaluated using different data splits, resolutions, or training protocols.

## 🧬 Dataset & Experimental Setup

| Item | Description |
|---|---|
| Dataset | MoNuSeg 2018 |
| Image type | H&E-stained histopathology |
| Training/validation | 37 images: 32 training + 5 validation |
| Held-out test set | 14 images |
| Test annotations | 6,697 ground-truth nuclei |
| Model family | U-Net-based segmentation |
| Final model | Boundary- and distance-aware multi-task U-Net |
| Input/crop resolution | 256 × 256 |

**Dataset note:** MoNuSeg data is not included in this repository. Obtain it from the official dataset source and comply with its terms of use.

## 🗂️ Explore the Repository

<details>
<summary><strong>📁 View project structure</strong></summary>

    NUCLEI-SEGMENTATION/
    ├── stages/
    │   ├── stage-01-target-generation/
    │   ├── stage-02-baseline-unet/
    │   ├── stage-03-boundary-aware-unet/
    │   ├── stage-04-multi-task-unet/
    │   └── stage-05-inference-evaluation/
    ├── docs/
    │   ├── methodology.md
    │   └── results.md
    ├── results/
    │   └── figures/
    ├── paper/
    │   └── README.md
    ├── reports/
    ├── requirements.txt
    ├── LICENSE
    └── README.md

</details>

### 📚 Navigate the Research

| Explore | Description |
|---|---|
| [🔬 Implementation stages](stages/) | Stage-wise project artifacts and notes |
| [📖 Methodology](docs/methodology.md) | Research workflow and model methodology |
| [📈 Results](docs/results.md) | Experimental metrics and findings |
| [📝 Research paper](paper/README.md) | Manuscript details and publication status |
| [📂 Project reports](reports/README.md) | Project documentation and reports |
| [🖼️ Figures](results/figures/) | Research figures (being organized) |

## ⚙️ Reproducibility & Code Status

The repository is a research record of the current project snapshot, not yet a one-command reproduction package.

- Exported HTML artifacts are present for Stages 1, 4 and 5.
- Python source files are present for Stage 2.
- Stage 3 source files were not present in the inspected repository snapshot; its stage folder documents this gap.
- Some scripts use local dataset paths that must be configured for the environment in which they are run.
- Dataset files are not redistributed here.

Start with the README inside the relevant stage directory and review the methodology before running an artifact.

## 🧭 Research Scope

**Implemented in this study**
- Multi-stage U-Net experimentation.
- Nucleus, boundary and distance-map prediction.
- Marker-controlled watershed reconstruction.
- Pixel-level and instance-level evaluation.

**Potential future directions**
- Morphological feature analysis.
- Spatial graph construction.
- Graph-neural-network-based cancer classification.

The future directions above are research opportunities, not completed features of this repository.

## 📄 Research Publication

**Boundary and Distance-Aware Multi-Task U-Net for Nuclei Instance Segmentation in Histopathological Images**

**Authors:** Meenakshi G, Kabel B, and Dayalin Leena G  
**Conference:** ICCWN 2026  
**Status:** Forthcoming — November 2026; official proceedings details and DOI are pending.

Publication metadata will be updated after the official proceedings record is available.

## 🌸 A Small Note on the Bigger Picture

A tissue section holds thousands of tiny cellular structures. Making those structures easier to distinguish is one step toward more useful computational analysis of histopathology images.

<div align="center">

**From tissue patterns to individual nuclei — one boundary at a time.** 🧬

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:e6a0ad,45:9d426b,100:160b18&height=120&section=footer&animation=fadeIn"/>

</div>

---

<div align="center">
<sub>Research project · Digital Pathology · Deep Learning · Nuclei Instance Segmentation</sub>
</div>
