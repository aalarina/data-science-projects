# Satellite Image Matching using LoFTR

## Overview

This project implements a satellite image matching pipeline for Sentinel-2 imagery using the pretrained **LoFTR** feature matching model.

The objective is to identify reliable correspondences between two satellite images acquired during different seasons without additional model training.

---

## Design choices

### Choice of Sentinel-2 True Color Images

The original dataset contains multiple Sentinel-2 spectral bands. However, this project utilizes only the True Color Image (TCI) representation.

The primary objective of the task is image matching rather than land cover classification or multispectral analysis. Since LoFTR is a feature matching model originally designed for RGB imagery, using TCI images provides a natural input format without requiring any architectural modifications.

### Confidence-based filtering

LoFTR produces a confidence score for every detected correspondence.

Instead of accepting every predicted match, only correspondences above a predefined confidence threshold are retained.

## Solution Explanation

The proposed solution consists of four main stages.

### 1. Dataset 

The project uses the **Deforestation in Ukraine** dataset from Kaggle.

The dataset contains Sentinel-2 Level-1C satellite imagery acquired between **2016 and 2019**.

Only **True Color Images (TCI)** stored as **JPEG2000 (.jp2)** files are used for matching.

Dataset source:
https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine

### Dataset preparation

The dataset preparation pipeline performs the following steps:

1. Locate all Sentinel-2 TCI images.
2. Extract acquisition dates and Sentinel tile identifiers.
3. Group images by Sentinel tile.
4. Select seasonal image pairs separated by approximately **180 ± 30 days**.
5. Copy the selected image pairs into a dedicated dataset directory.
6. Store metadata for every image pair.

Each prepared sample has the following structure:

```
pair_YYYYMMDD_YYYYMMDD/

    image_A.jp2

    image_B.jp2

    metadata.json
```


### 2. Image Preprocessing

Each Sentinel-2 image has a spatial resolution of **10980 × 10980 pixels**, which exceeds the available GPU memory during inference. 

Instead of resizing the images, every scene is divided into **1024 × 1024** non-overlapping tiles. 

This approach preserves the original spatial resolution while allowing efficient inference.


### 3. Image Matching

Each corresponding image tile is independently processed by the pretrained **LoFTR** model.

The detected local correspondences are transformed into global image coordinates before being merged into a single set of matches.


### 4. Post-processing

The resulting correspondences are filtered according to their confidence score.

Only high-confidence matches are retained and visualized.

---

## Why LoFTR?

The pretrained **LoFTR Outdoor** model was selected for several reasons:

- detector-free feature matching;
- robustness to significant appearance changes;
- state-of-the-art performance on outdoor image matching benchmarks;
- publicly available pretrained weights;
- suitability for large-scale satellite imagery without additional training.

Unlike traditional feature detectors such as SIFT or ORB, LoFTR directly predicts dense correspondences using Transformer-based local feature aggregation, making it more robust to seasonal appearance variations.

---

## Model weights

The project uses the pretrained **LoFTR Outdoor** model available from Kornia.

Weights:
- https://huggingface.co/kornia/loftr/blob/main/loftr_outdoor.ckpt

The weights are downloaded automatically when executing:

```python
matcher = KF.LoFTR(pretrained="outdoor")
```

Therefore, no manual download of the model weights is required.

---

## Project Structure

```text
Satellite-Image-Matching/
│
├── notebooks/
│   ├── dataset_creation.ipynb     # Dataset preparation from Sentinel-2 images
│   └── demo_inference.ipynb       # End-to-end inference demonstration
│
├── src/
│   ├── algorithm.py               # LoFTR matching pipeline
│   └── inference.py               # Visualization and inference utilities
│
├── README.md                      
└── requirements.txt               
```

---

## Project Setup

Clone the repository

```bash
git clone <repository_url>

cd <repository_name>
```

Install all required packages

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run

```bash
python inference.py
```

The script automatically:

- loads one image pair;
- splits both images into tiles;
- performs LoFTR matching;
- filters low-confidence correspondences;
- visualizes the strongest matches.

---

## Experimental results

Two seasonal image pairs were evaluated.

### Example 1

**12 February 2016 <--> 30 August 2016**

(Winter <--> Summer)

After confidence filtering:

| Metric | Value |
|-------|------:|
| Number of correspondences | **45,831** |
| Mean confidence | **0.9106** |
| Maximum confidence | **0.9998** |

This example represents a challenging winter-to-summer scenario with extensive snow cover. Despite significant appearance changes, the proposed pipeline identifies reliable correspondences between stable landscape structures.

---

### Example 2

**18 March 2019 <--> 15 August 2019**

(Spring <--> Summer)

After confidence filtering:

| Metric | Value |
|-------|------:|
| Number of correspondences | **185,448** |
| Mean confidence | **0.9354** |
| Maximum confidence | **0.99999** |

Compared to the winter example, this image pair contains substantially more visual texture, resulting in a larger number of reliable correspondences and higher average confidence.

Image comparing results are available in ```inference_demo.ipynb```.

---

## Discussion

The proposed pipeline successfully matches Sentinel-2 images acquired during different seasons using the pretrained LoFTR Outdoor model.

The experimental evaluation demonstrates that the model remains robust to seasonal appearance changes, including variations in vegetation, illumination, and partial snow cover, without requiring additional training or fine-tuning.

The spring-to-summer image pair produced substantially more reliable correspondences than the winter-to-summer pair, which can be explained by the greater amount of visual texture and fewer homogeneous regions. In contrast, extensive snow cover reduces the number of distinctive features available for matching, leading to fewer high-confidence correspondences.

Overall, the obtained results indicate that the proposed pipeline is suitable for feature matching in Sentinel-2 imagery under varying seasonal conditions.

---

## Limitations

The current implementation has several limitations:

- images are divided into fixed **1024 × 1024** non-overlapping tiles
- image tiles are processed independently without interaction between neighboring tiles
- no geometric verification (e.g., RANSAC) is applied after LoFTR matching
- cloud-contaminated regions are not explicitly removed before matching

---

## Possible improvements

Several improvements could further increase the matching quality:

- use overlapping tiles to reduce boundary effects
- apply geometric verification (e.g., RANSAC) to remove incorrect correspondences
- fine-tune LoFTR
- use cloud masks to exclude unreliable regions before keypoint detection
- implement adaptive tile sizes depending on available GPU memory
- benchmark alternative feature matching models (e.g., LightGlue, SuperGlue and EfficientLoFTR) to identify the most effective solution for Sentinel-2 seasonal image matching
