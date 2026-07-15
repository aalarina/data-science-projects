"""
Satellite image matching algorithm based on the pretrained LoFTR model.

Pipeline:
1. Load Sentinel-2 image pair.
2. Split images into tiles.
3. Match each tile using LoFTR.
4. Transform local coordinates into global image coordinates.
5. Merge all matches.
"""

from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import kornia.feature as KF
from tqdm import tqdm


# ============================================================
# Image loading
# ============================================================

def read_rgb(path: Path):
    """
    Read Sentinel-2 TCI image.

    Parameters
    ----------
    path : Path

    Returns
    -------
    numpy.ndarray
        RGB image.
    """

    with rasterio.open(path) as src:
        image = src.read([1, 2, 3])

    image = np.moveaxis(image, 0, -1)

    return image


# ============================================================
# Image tiling
# ============================================================

def split_image_into_tiles(image, tile_size=1024):
    """
    Split image into non-overlapping tiles.

    Parameters
    ----------
    image : ndarray
    tile_size : int

    Returns
    -------
    list
    """

    h, w = image.shape[:2]

    tiles = []

    for y in range(0, h - tile_size + 1, tile_size):

        for x in range(0, w - tile_size + 1, tile_size):

            tiles.append({
                "image": image[y:y+tile_size, x:x+tile_size],
                "x": x,
                "y": y
            })

    return tiles


# ============================================================
# LoFTR matching
# ============================================================

def match_tile_pair(tileA, tileB, matcher, device):
    """
    Match one tile pair using LoFTR.
    """

    grayA = cv2.cvtColor(tileA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(tileB, cv2.COLOR_RGB2GRAY)

    tensorA = torch.from_numpy(grayA).float() / 255.
    tensorA = tensorA.unsqueeze(0).unsqueeze(0).to(device)

    tensorB = torch.from_numpy(grayB).float() / 255.
    tensorB = tensorB.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():

        correspondences = matcher({

            "image0": tensorA,

            "image1": tensorB

        })

    return correspondences


# ============================================================
# Whole image matching
# ============================================================

def match_image_pair(tilesA, tilesB, matcher, device):
    """
    Match all corresponding tiles.

    Returns
    -------
    keypoints0
    keypoints1
    confidence
    """

    all_keypoints0 = []
    all_keypoints1 = []
    all_confidence = []

    for tileA, tileB in tqdm(
            zip(tilesA, tilesB),
            total=len(tilesA),
            desc="Matching tiles"):

        corr = match_tile_pair(
            tileA["image"],
            tileB["image"],
            matcher,
            device
        )

        kp0 = corr["keypoints0"].cpu().numpy()
        kp1 = corr["keypoints1"].cpu().numpy()
        conf = corr["confidence"].cpu().numpy()

        # Convert local tile coordinates
        # into global image coordinates

        kp0[:, 0] += tileA["x"]
        kp0[:, 1] += tileA["y"]

        kp1[:, 0] += tileB["x"]
        kp1[:, 1] += tileB["y"]

        all_keypoints0.append(kp0)
        all_keypoints1.append(kp1)
        all_confidence.append(conf)

    return (
        np.concatenate(all_keypoints0),
        np.concatenate(all_keypoints1),
        np.concatenate(all_confidence)
    )


# ============================================================
# Main algorithm
# ============================================================

def run_algorithm(pair_folder: Path):
    """
    Execute the complete image matching pipeline.
    """

    imageA = read_rgb(pair_folder / "image_A.jp2")
    imageB = read_rgb(pair_folder / "image_B.jp2")

    tilesA = split_image_into_tiles(imageA)
    tilesB = split_image_into_tiles(imageB)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    matcher = KF.LoFTR(pretrained="outdoor").to(device)
    matcher.eval()

    keypoints0, keypoints1, confidence = match_image_pair(
        tilesA,
        tilesB,
        matcher,
        device
    )

    return keypoints0, keypoints1, confidence


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    dataset = Path("prepared_dataset")

    pair = sorted(dataset.iterdir())[0]

    kp0, kp1, conf = run_algorithm(pair)

    print(f"Total matches: {len(conf)}")
