# EQLv3 with CBNet: Enhanced Object Detection Models

## Overview

This repository is the official implementation of ***["AAMC: Annealing the Attention of Minority Classes for Imbalanced Learning"]()***.

If you found this project useful, please give us a star ⭐️ or [cite]() us in your paper, this is the greatest support and encouragement for us.

## Environment Setup

### Prerequisites

- Python 3.7+
- CUDA 11.0+ (for GPU training)
- Pytorch 1.12.1
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html) 
- mmengine
- mmcv<2.0.0

### Installation

1. **Create a conda environment**:
```bash
conda create -n aamc python=3.8
conda activate aamc
```

2. **Install PyTorch 1.12.1**:

 Install [PyTorch](https://pytorch.org/get-started/previous-versions/#v212). If you have experience with PyTorch and have already installed it, you can skip to the next section.

3. **Install MMEngine, MMCV, and MMDetection using MIM.**:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv<2.0.0"
mim install mmdet
```

4. **Install other dependencies from requirements.txt**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### LVIS Dataset

1. **Download [LVIS V1]([LVIS](https://www.lvisdataset.org/dataset))dataset**:

​	[LVIS V1]([LVIS](https://www.lvisdataset.org/dataset))

​	Please download the v1.0 Training set and Validation set.

2. **Organize dataset structure**:
```
data/
├── lvis_v1/
│   ├── annotations/
│   │   ├── lvis_v1_train.json
│   │   └── lvis_v1_val.json
│   └── images/
│       ├── train2017/
│       └── val2017/
```

## Training

### Single GPU Training

```bash
# Train ResNet50 + CBNet + EQLv3 model
python tools/train.py configs/cbnet2/new_eqlv3[9.5-1.5]-sample1e-3_sgd_thr1e-4_r50_8x2_3x.py

# Train Swin Transformer + CBNet + EQLv3 model
python tools/train.py configs/cbnet2/eqlv3_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_lvis-thr-0_a-8-0.py
```

### Multi-GPU Training

```bash
# Train with 8 GPUs
bash tools/dist_train.sh configs/cbnet2/new_eqlv3[9.5-1.5]-sample1e-3_sgd_thr1e-4_r50_8x2_3x.py 8

# Train Swin model with 8 GPUs
bash tools/dist_train.sh configs/cbnet2/eqlv3_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_lvis-thr-0_a-8-0.py 8
```

## Testing

### Single GPU Testing

```bash
# Test ResNet50 model
python tools/test.py configs/cbnet2/new_eqlv3[9.5-1.5]-sample1e-3_sgd_thr1e-4_r50_8x2_3x.py \
    work_dirs/new_eqlv3[9.5-1.5]-sample1e-3_sgd_thr1e-4_r50_8x2_3x/latest.pth \
    --eval bbox segm

# Test Swin Transformer model
python tools/test.py configs/cbnet2/eqlv3_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_lvis-thr-0_a-8-0.py \
    work_dirs/eqlv3_mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_lvis-thr-0_a-8-0/latest.pth \
    --eval bbox segm
```

### Multi-GPU Testing

```bash
# Test with 8 GPUs
bash tools/dist_test.sh configs/cbnet2/new_eqlv3[9.5-1.5]-sample1e-3_sgd_thr1e-4_r50_8x2_3x.py \
    work_dirs/new_eqlv3[9.5-1.5]-sample1e-3_sgd_thr1e-4_r50_8x2_3x/latest.pth \
    8 --eval bbox segm
```

## Acknowledgments

- [MMDetection](https://github.com/open-mmlab/mmdetection) for the detection framework
- [CBNet](https://arxiv.org/abs/2107.00420) for the composite backbone architecture
- [EQLv2](https://arxiv.org/abs/2012.08548) for the equalization loss foundation
- [Swin Transformer](https://arxiv.org/abs/2103.14030) for the transformer backbone
