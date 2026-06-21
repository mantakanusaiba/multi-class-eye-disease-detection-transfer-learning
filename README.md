# Multi-Class Eye Disease Detection Using Transfer Learning

A comparative study of three pretrained CNN architectures — **ResNet-50**, **DenseNet-121**, and **EfficientNet-B0** — for four-class retinal fundus image classification, evaluated **with and without** a CLAHE-based retinal segmentation preprocessing pipeline.

This repository contains the training notebooks, the project report, and the dataset reference used in the study *"Multi-Class Eye Disease Detection Using Transfer Learning: A Comparative Study with and without Retinal Segmentation."*

## Authors

Mantaka Nusaiba, Atandrila Pushpa, Mahia Rahman — Department of Computer Science and Engineering, Ahsanullah University of Science and Technology (AUST), Dhaka, Bangladesh

## Overview

Eye diseases such as Diabetic Retinopathy, Glaucoma, and Cataract are leading causes of preventable vision loss worldwide. This project investigates whether a hand-crafted segmentation pipeline (blood vessel extraction, optic disc detection, and exudate isolation) meaningfully improves transfer-learning classification performance compared to a standard augmentation-only pipeline, across three different CNN backbones.

All six configurations share an identical classification head (Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)) and are trained on the same stratified train/validation/test split to isolate the effect of preprocessing from architecture choice.

## Dataset

- **Source:** [Eye Diseases Classification dataset on Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) (see `Dataset_Link.txt`)
- **Size:** 4,217 RGB retinal fundus images
- **Classes:** Normal, Cataract, Glaucoma, Diabetic Retinopathy
- **Split:** 70% train / 15% validation / 15% test (stratified)

The dataset itself is not included in this repository due to size and licensing — download it directly from Kaggle using the link above.

## Repository Structure

```
.
├── README.md
├── Dataset_Link.txt                                 # Kaggle dataset source
└── notebooks/
    ├── resnet50_without_segmentation.ipynb
    ├── resnet50_with_segmentation.ipynb
    ├── densenet121_without_segmentation.ipynb
    ├── densenet121_with_segmentation.ipynb
    ├── efficientnetb0_without_segmentation.ipynb
    └── efficientnetb0_with_segmentation.ipynb
```

## Methodology

### Preprocessing Pipelines

**Without Segmentation:** Images resized to 224×224, normalized with the architecture-specific `preprocess_input` function, augmented with random horizontal flipping.

**With Segmentation:** In addition to the above, each image passes through:
1. **CLAHE enhancement** — LAB color space, applied to the luminance channel
2. **Blood vessel extraction** — green channel, Gaussian blur, Otsu thresholding, morphological opening
3. **Optic disc segmentation** — Hough Circle Transform on the median-blurred grayscale image
4. **Exudate detection** — Otsu thresholding + morphological opening after masking the optic disc
5. **Mask combination** — the three binary masks are merged and applied to the CLAHE-enhanced image to suppress background regions

### Models

All backbones are initialized with ImageNet weights and fine-tuned with a shared classification head (GAP → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)).

| Model | Unfrozen Layers (w/o Seg.) | Unfrozen Layers (w/ Seg.) | Epochs |
|---|---|---|---|
| ResNet-50 | Last 30 | Last 30 | 15 / 10 |
| DenseNet-121 | Last 30 | Last 40 | 15 / 10 |
| EfficientNet-B0 | Last 30 | Last 30 | 15 / 10 |

Optimizer: Adam · Loss: categorical cross-entropy · Batch size: 8

## Results

| Model | Preprocessing | Test Accuracy (%) | Macro F1 |
|---|---|---|---|
| ResNet-50 | w/o Segmentation | 92.41 | 0.9220 |
| ResNet-50 | w/ Segmentation | 92.72 | 0.9268 |
| DenseNet-121 | w/o Segmentation | 93.35 | 0.9329 |
| **DenseNet-121** | **w/ Segmentation** | **93.83** | **0.9374** |
| EfficientNet-B0 | w/o Segmentation | 93.20 | 0.9303 |
| EfficientNet-B0 | w/ Segmentation | 92.72 | 0.9258 |

**Key findings:**
- **DenseNet-121 + segmentation** achieves the best overall result at **93.83% test accuracy**.
- Segmentation improves accuracy for ResNet-50 and DenseNet-121, but slightly *reduces* it for EfficientNet-B0 — likely because its compound-scaling already captures sufficient low-level features.
- **Glaucoma** is consistently the hardest class to classify across all six configurations (F1: 0.857–0.893).
- **Diabetic Retinopathy** is the most reliably detected class, exceeding 0.99 F1 in five of six configurations.

Full per-class precision/recall/F1 tables, the radar chart comparison, and discussion are available in the accompanying project report.

## How to Run

1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) and place it according to the path expected at the top of each notebook (update the path if needed).
2. Open the relevant notebook in `notebooks/` (e.g. Jupyter, Google Colab, or Kaggle Notebooks).
3. Install dependencies — primarily `tensorflow`/`keras`, `opencv-python`, `scikit-learn`, `numpy`, and `matplotlib`.
4. Run all cells top to bottom. Each notebook trains and evaluates a single (architecture × preprocessing) configuration.

## Limitations

- Dataset size (4,217 images) is modest for deep learning, particularly for the Glaucoma class.
- The segmentation pipeline uses classical CV techniques (Otsu, Hough Circles) sensitive to image quality and lighting.
- Only horizontal flipping is used as augmentation.
- Study limited to three architectures; ViT, MobileNetV3, and InceptionV3 are left for future work.

## Future Work

- Replace the hand-crafted segmentation pipeline with learned segmentation (U-Net, DeepLabV3+).
- Add interpretability via attention maps / class activation maps.
- Extend to fine-grained Diabetic Retinopathy severity grading.
- Deploy as a web-based screening tool for low-resource clinical settings.
