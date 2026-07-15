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

def read_rgb(image_path):
    """
    Read Sentinel-2 RGB image (TCI.jp2).

    Parameters
    ----------
    image_path --- Path to the RGB Sentinel-2 image.

    Returns
    -------
    numpy.ndarray --- RGB image as uint8 numpy array.
    """

    with rasterio.open(image_path) as src:
        image = src.read()

    image = np.moveaxis(image, 0, -1)

    return image


# ============================================================
# Image tiling
# ============================================================

def split_into_tiles(image, tile_size=1024):
    """
    Split image into non-overlapping tiles.

    Parameters
    ----------
    image --- Input RGB image.
    tile_size --- Tile size in pixels.

    Returns
    -------
    list --- List of dictionaries.

        Each dictionary contains:
        {
            "image": tile,
            "x": x_offset,
            "y": y_offset
        }
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
# LoFTR initialization
# ============================================================

def load_matcher(device):
    """
    Load the pretrained LoFTR model.

    Parameters
    ----------
    device --- Device used for inference (CPU or CUDA).

    Returns
    -------
    KF.LoFTR --- Initialized LoFTR model.
    """

    matcher = KF.LoFTR(pretrained="outdoor").to(device)

    matcher.eval()

    return matcher


# ============================================================
# Tile matching
# ============================================================

def match_single_tile(tileA, tileB, matcher, device):
    """
    Match two image tiles using LoFTR.

    Parameters
    ----------
    tileA --- First RGB tile.
    tileB ---Second RGB tile.
    matcher --- Initialized LoFTR model.
    device --- Device used for inference.

    Returns
    -------
    dict --- Dictionary containing LoFTR correspondences.
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
# Full image matching
# ============================================================

def match_image_pair(tilesA, tilesB, matcher, device):
    """
    Match all corresponding image tiles.
    
    Each tile is processed independently using the pretrained
    LoFTR model. Local keypoints are converted into global
    image coordinates before being merged.

    Parameters
    ----------
    tilesA --- Tiles from the first image.
    tilesB --- Tiles from the second image.
    matcher --- Initialized LoFTR model.
    device --- Device used for inference.

    Returns
    -------
    tuple --- (keypoints0, keypoints1, confidence)
    """

    all_keypoints0 = []
    all_keypoints1 = []
    all_confidence = []

    for tileA, tileB in tqdm(
            zip(tilesA, tilesB),
            total=len(tilesA),
            desc="Matching image tiles"):

        correspondences = match_single_tile(
            tileA["image"],
            tileB["image"],
            matcher,
            device
        )

        kp0 = correspondences["keypoints0"].cpu().numpy()
        kp1 = correspondences["keypoints1"].cpu().numpy()
        conf = correspondences["confidence"].cpu().numpy()

        # Skip tiles where no correspondences were found
        if len(conf) == 0:
            continue
            
        # Convert local tile coordinates
        # into global image coordinates

        kp0[:, 0] += tileA["x"]
        kp0[:, 1] += tileA["y"]

        kp1[:, 0] += tileB["x"]
        kp1[:, 1] += tileB["y"]

        all_keypoints0.append(kp0)
        all_keypoints1.append(kp1)
        all_confidence.append(conf)

    keypoints0 = np.concatenate(all_keypoints0)
    keypoints1 = np.concatenate(all_keypoints1)
    confidence = np.concatenate(all_confidence)

    return keypoints0, keypoints1, confidence


# ============================================================
# Main algorithm
# ============================================================

def run_matching(imageA_path, imageB_path, tile_size = 1024):
    """
    Run the complete Sentinel-2 image matching pipeline.

    Pipeline:
        - Read two RGB Sentinel-2 images.
        - Split both images into tiles.
        - Match every tile using LoFTR.

    Parameters
    ----------
    imageA_path --- Path to the first image.
    imageB_path --- Path to the second image.
    tile_size --- Tile size used during matching.

    Returns
    -------
    tuple
        imgA 
        imgB 
        keypoints0 
        keypoints1 
        confidence
    """

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------
    # Load pretrained LoFTR
    # --------------------------------------------------------

    matcher = load_matcher(device)

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    imgA = read_rgb(imageA_path)
    imgB = read_rgb(imageB_path)

    # --------------------------------------------------------
    # Split images into tiles
    # --------------------------------------------------------

    tilesA = split_into_tiles(imgA, tile_size)
    tilesB = split_into_tiles(imgB, tile_size)

    # --------------------------------------------------------
    # Match all tiles
    # --------------------------------------------------------

    keypoints0, keypoints1, confidence = match_all_tiles(
        tilesA,
        tilesB,
        matcher,
        device
    )

    return (imgA, imgB, keypoints0, keypoints1, confidence)
