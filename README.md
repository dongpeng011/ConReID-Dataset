# Robust Person Re-identification in Complex Construction Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official repository for the paper: **"Robust Person Re-identification in Complex Construction Environments via Implicit Scene-Adaptive Prompting and Geometric Manifold Alignment"**. 

This repository also hosts the newly constructed **Con-ReID dataset**, which is specifically designed for person Re-identification (ReID) in complex industrial and construction environments.

---

## 1. Introduction

Conventional ReID models encounter significant limitations in construction sites. These environments feature severe visual similarity (identical safety uniforms), low-resolution targets, massive visual blind spots, and extreme environmental dynamics. 

To overcome these challenges without relying on manual scene labels, we propose a unified framework named **ISGA-ViT**. It employs a cohesive, multi-layer strategy to extract reliable identity features in dynamic construction environments.

### Key Components of ISGA-ViT:
* **SFC (Semantic Feature Calibration):** Acts as a feature denoiser. It filters out the visual similarity trap caused by identical safety vests and forces the model to focus on unique identity details.
* **LCP (Latent Context Prompting):** Operates entirely without manual labels. It implicitly infers specific scene contexts (e.g., occlusion, infrared) and generates adaptive prompts to adjust the Transformer's attention in real-time.
* **GMA (Geometric Manifold Alignment):** Integrates learnable camera and viewpoint embeddings to explicitly correct extreme geometric distortions found in irregular camera networks.

---

## 2. The Con-ReID Dataset

Existing datasets mostly focus on city streets or campuses. To bridge the gap for industrial scenarios, we constructed **Con-ReID**. It is collected directly from real-world operating construction sites.

### Dataset Statistics:
* **Total Images:** 13,130
* **Total Identities:** 154
* **Cameras:** 6 (covering front, back, side, and top-down views)
* **Image Size:** Standardized to 64 × 128 pixels (with Zero-Padding to preserve aspect ratio).
* **Challenges Included:** Heavy occlusion, low-resolution (distant targets), and cross-modality (pseudo-infrared) images.

### Data Splitting:
| Subset | Identities | Images | Description |
| :--- | :--- | :--- | :--- |
| **Train Set** | 91 | 8,844 | Used to optimize model parameters. |
| **Query Set** | 30 | 102 | Targets to be re-identified. |
| **Gallery Set** | 33 | 4,184 | Reference database. |

### Naming Convention:
To ensure compatibility with existing evaluation codes, we strictly follow the Market-1501 naming format:
`ID_cXsY_ZZZZZZ.jpg`
*(e.g., `0002_c5s1_000001.jpg` means Identity 0002, Camera 5, Sequence 1, Frame 000001).*

**[🔗 Click here to download the Con-ReID dataset ((https://github.com/dongpeng011/ConReID-Dataset))]**

---

## 3. Main Results

Our ISGA-ViT framework achieves state-of-the-art (SOTA) performance on multiple public benchmarks and the Con-ReID dataset under the Multi-Scene Joint Training setting.

| Dataset | mAP (%) | Rank-1 (%) |
| :--- | :--- | :--- |
| **Con-ReID** | **89.6** | **97.3** |
| Market1501 | 80.6 | 92.3 |
| Msmt17 | 42.1 | 66.2 |
| MLR-CUHK03 | 87.2 | 91.9 |
| Occ-Duke | 59.7 | 72.9 |

> *For more detailed results, including ablation studies and single-scene performance, please refer to our paper.*

---

## 4. Getting Started

### Prerequisites
* Linux (Ubuntu 22.04 recommended)
* Python >= 3.8
* PyTorch >= 2.0 (CUDA 12.1 recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/dongpeng011/ISGA-ViT.git
cd ISGA-ViT

# Install dependencies
pip install -r requirements.txt
ISGA-ViT/
├── data/
│   └── Con-ReID/
│       ├── bounding_box_train/
│       ├── bounding_box_test/
│       └── query/
If you have any questions, please feel free to open an issue or contact liuhongbin19@sdjzu.edu.cn.
