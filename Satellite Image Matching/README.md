# Satellite Image Matching using LoFTR

## Overview

This project implements a satellite image matching pipeline for Sentinel-2 imagery using the pretrained **LoFTR** feature matching model.

The objective is to identify reliable correspondences between two satellite images acquired during different seasons without additional model training.

---

## Solution Explanation

## Dataset

The project uses the **Deforestation in Ukraine** dataset from Kaggle.

The dataset contains Sentinel-2 Level-1C satellite imagery acquired between **2016 and 2019**.

Only **True Color Images (TCI)** stored as **JPEG2000 (.jp2)** files are used for matching.

Dataset source:
https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine

---

## Dataset preparation

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

---

## Matching pipeline

The proposed image matching pipeline consists of the following stages:

1. Read Sentinel-2 RGB images.
2. Split large images into **1024 × 1024** non-overlapping tiles.
3. Perform feature matching independently on every tile using the pretrained LoFTR model.
4. Transform local tile coordinates into global image coordinates.
5. Merge all detected correspondences.
6. Filter correspondences by confidence score.
7. Visualize the strongest matches.

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

## Running the project

The script automatically:

- loads one image pair
- runs the LoFTR matching pipeline
- filters low-confidence correspondences
- visualizes the strongest matches

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

---

## Conclusion

A complete image matching pipeline for Sentinel-2 satellite imagery was successfully implemented using the pretrained LoFTR Outdoor model.
The project includes dataset preparation, tile-based matching, confidence-based filtering and visualization of the detected correspondences.

The obtained results demonstrate that reliable image matching is possible even under significant seasonal appearance changes, while also highlighting the influence of snow cover and low-texture regions on the number of detected correspondences.

The proposed implementation provides a solid baseline for future research and can be further extended with more advanced post-processing techniques or alternative deep feature matching models.
