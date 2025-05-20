# Enhanced DiffusionDet: Improving Object Detection in Dense Scenes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: MMDetection](https://img.shields.io/badge/Framework-MMDetection%203.3.0-blue)](https://github.com/open-mmlab/mmdetection)
[![Python: 3.7+](https://img.shields.io/badge/Python-3.7%2B-green)](https://www.python.org/)

This repository contains an enhanced version of DiffusionDet, designed to improve detection performance in dense object scenarios. The implementation is based on MMDetection 3.3.0 framework.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Evaluation](#evaluation)
- [References](#references)

## Introduction

DiffusionDet is a diffusion-based object detection model that shows promising results. However, it faces challenges in dense scenes with similar objects. Our enhanced version addresses these issues through three specialized modules:

1. **Local Affinity Module (LAM)** - Enhances feature consistency in local regions
2. **Local Collaborating Module (LCM)** - Promotes information exchange between nearby objects
3. **Auxiliary Density Estimation Module (ADEM)** - Provides density awareness to improve detection

## Key Features

- **Improved detection in dense scenes** with similar-looking objects
- **Enhanced feature consistency** through local affinity modeling
- **Better context understanding** via local collaborative mechanisms
- **Density-aware detection** with auxiliary density estimation
- **Fully compatible** with the original MMDetection framework
- **Dynamic DDIM sampling** that adapts to image complexity
- **Small object distillation** for better small object detection

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+
- MMDetection 3.3.0

### Setup

1. Install MMDetection following the [official guide](https://mmdetection.readthedocs.io/en/latest/get_started.html)

2. Clone this repository:
   ```bash
   git clone https://github.com/ZivNice/algaeDiff-net.git
   cd algaeDiff-net
   ```

3. Copy the implementation files to your MMDetection project:
   ```bash
   # Copy model files
   cp -r mmdet/models/necks/lam_fpn.py /path/to/mmdetection/mmdet/models/necks/
   cp -r mmdet/models/dense_heads/adem_head.py /path/to/mmdetection/mmdet/models/dense_heads/
   cp -r mmdet/models/utils/lcm_modules.py /path/to/mmdetection/mmdet/models/utils/
   
   # Copy DiffusionDet enhancement files
   cp -r projects/DiffusionDet/diffusiondet/* /path/to/mmdetection/projects/DiffusionDet/diffusiondet/
   
   # Copy evaluation metrics
   cp -r mmdet/evaluation/metrics/density_coco_metric.py /path/to/mmdetection/mmdet/evaluation/metrics/
   ```

4. Update the corresponding `__init__.py` files to register the new modules

## Usage

### Training

```bash
# Single GPU
python tools/train.py configs/diffusiondet/diffusiondet_r50_lam_lcm_adem.py

# Multiple GPUs
bash tools/dist_train.sh configs/diffusiondet/diffusiondet_r50_lam_lcm_adem.py 8
```

### Inference

```bash
# Single GPU
python tools/test.py configs/diffusiondet/diffusiondet_r50_lam_lcm_adem.py /path/to/checkpoint.pth

# Multiple GPUs
bash tools/dist_test.sh configs/diffusiondet/diffusiondet_r50_lam_lcm_adem.py /path/to/checkpoint.pth 8
```

## Model Architecture

Our enhanced DiffusionDet builds upon the original architecture with three key modules:

### 1. Local Affinity Module (LAM)

<details>
<summary>Click to expand details</summary>

The LAM module captures common attributes within local regions to enhance feature consistency:

- **Input**: Features from two adjacent FPN levels
- **Process**: 
  1. Concatenates features from both levels
  2. Generates an affinity map using convolutions
  3. Uses the affinity map to weight and combine features
- **Output**: Enhanced features with improved local consistency
- **Benefits**: Helps suppress noise and highlight common features of similar objects

</details>

### 2. Local Collaborating Module (LCM)

<details>
<summary>Click to expand details</summary>

The LCM module enhances information exchange between nearby objects:

- **Input**: ROI features from detection head
- **Process**:
  1. Maps ROI features to a consistency space
  2. Predicts offsets for box refinement
  3. Applies guidance scale to control refinement strength
- **Output**: Refined bounding box coordinates
- **Benefits**: Improves detection accuracy in crowded scenes

</details>

### 3. Auxiliary Density Estimation Module (ADEM)

<details>
<summary>Click to expand details</summary>

The ADEM module provides density awareness to improve detection:

- **Input**: FPN features
- **Process**:
  1. Processes features through scale-specific processors
  2. Generates a density map and uncertainty estimates
  3. Uses density information to enhance detection confidence
- **Output**: Density map and uncertainty estimates
- **Benefits**: Helps model understand object distribution patterns

</details>

## Implementation Details

### Code Structure

```
mmdetection/
├── configs/diffusiondet/
│   └── diffusiondet_r50_lam_lcm_adem.py  # Main configuration file
├── mmdet/models/
│   ├── necks/
│   │   └── lam_fpn.py                   # LAM enhanced FPN
│   ├── dense_heads/
│   │   └── adem_head.py                 # ADEM module
│   └── utils/
│       └── lcm_modules.py               # LCM core module
├── projects/DiffusionDet/
│   ├── diffusiondet/
│   │   ├── diffusiondet_enhanced_head.py  # Enhanced detection head 
│   │   ├── loss.py                      # Extended loss calculation
│   │   ├── small_object_distill.py      # Small object distillation
│   │   └── datasets/
│   │       └── loading.py               # Density map generation
└── mmdet/evaluation/metrics/
    └── density_coco_metric.py           # Density-aware evaluation
```

### Key Integration Points

1. **LAMFPN**: Enhances feature pyramids with spatial-channel attention
2. **LCM**: Embedded in the detection head for feature refinement
3. **ADEM**: Provides density estimation to improve candidate box generation
4. **Dynamic DDIM**: Adapts sampling steps based on image complexity
5. **Small Object Distillation**: Focuses knowledge distillation on small objects

## Evaluation

Our model is evaluated using the standard COCO metrics, plus our custom density-aware metrics that categorize test images into:

- **Sparse scenes**: ≤3 objects
- **Normal scenes**: 4-15 objects
- **Dense scenes**: >15 objects

This provides a more detailed understanding of model performance across different scene complexities.

## References

- [DiffusionDet: Diffusion Model for Object Detection](https://arxiv.org/abs/2211.09788)
- [MMDetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{enhanced-diffusiondet,
  author = {Your Name},
  title = {AlgaeDiff-Net: Integrating Density Estimation with Diffusion Models for Enhanced Microalgae Detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/enhanced-diffusiondet}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
