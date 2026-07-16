# Satellite Image Matching using LoFTR

## Overview

This project implements a satellite image matching pipeline for Sentinel-2 imagery using the pretrained **LoFTR** feature matching model.

The objective is to identify reliable correspondences between two satellite images acquired during different seasons without additional model training.

---

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

## Running the project

The script automatically:

- loads one image pair,
- runs the LoFTR matching pipeline,
- filters low-confidence correspondences,
- visualizes the strongest matches.

---

## Experimental results

Two seasonal image pairs were evaluated.

### Example 1

**12 February 2016 --> 30 August 2016**

(Winter --> Summer)

After confidence filtering:

| Metric | Value |
|-------|------:|
| Number of correspondences | **45,831** |
| Mean confidence | **0.9106** |
| Maximum confidence | **0.9998** |

This example represents a challenging winter-to-summer scenario with extensive snow cover. Despite significant appearance changes, the proposed pipeline successfully identifies reliable correspondences between stable landscape structures.

---

### Example 2

**18 March 2019 --> 15 August 2019**

(Spring --> Summer)

After confidence filtering:

| Metric | Value |
|-------|------:|
| Number of correspondences | **185,448** |
| Mean confidence | **0.9354** |
| Maximum confidence | **0.99999** |

Compared to the winter example, this image pair contains substantially more visual texture, resulting in a larger number of reliable correspondences and higher average confidence.

---

## Discussion

The pretrained LoFTR model demonstrates strong robustness to seasonal appearance changes without any additional training.

The algorithm successfully detects correspondences between Sentinel-2 images even when vegetation, illumination or snow cover differ considerably.

As expected, the matching quality decreases in homogeneous regions with limited visual texture, particularly in snow-covered areas.

---

## Limitations

Current limitations of the proposed pipeline include:

- fixed tile size (1024 × 1024)
- independent processing of image tiles
- no geometric verification (e.g., RANSAC) after LoFTR matching
- no fine-tuning of the pretrained model on Sentinel-2 imagery

---

## Possible improvements

Several improvements could further increase the matching quality:

- use overlapping tiles instead of non-overlapping ones
- apply geometric verification (e.g. RANSAC)
- fine-tune LoFTR on remote sensing imagery
- use cloud masks to exclude unreliable regions before keypoint detection
- introduce adaptive tile sizes for memory-efficient inference

Future work may include evaluating correspondence quality inside deforestation regions using the provided GeoJSON annotations.

---

## Conclusion

A complete satellite image matching pipeline based on the pretrained LoFTR model was successfully implemented.

The proposed approach supports Sentinel-2 imagery acquired during different seasons and demonstrates reliable matching performance without additional model training.

The modular implementation allows the pipeline to be easily extended with more advanced post-processing or domain-specific fine-tuning in future work.
