# ==================================================
# Imports
# ==================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.algorithm import run_matching


# ==================================================
# Configuration
# ==================================================

# Minimum LoFTR confidence required to keep a correspondence
CONFIDENCE_THRESHOLD = 0.8

# Maximum number of strongest correspondences to visualize
TOP_K = 100

# Index of the image pair from the prepared dataset
PAIR_INDEX = 0


# ==================================================
# Data loading
# ==================================================

# Load one image pair from the prepared dataset
pair = sorted(Path("/prepared_dataset").iterdir())[PAIR_INDEX]

imageA_path = pair / "image_A.jp2"
imageB_path = pair / "image_B.jp2"


# ==================================================
# Match filtering and visualization
# ==================================================

def filter_matches(keypoints0, keypoints1, confidence, threshold=0.8, top_k=100):
    """
    Filter LoFTR correspondences.

    Parameters
    ----------
    keypoints0 --- Keypoints detected in the first image.
    keypoints1 --- Corresponding keypoints in the second image.
    confidence --- Matching confidence scores.
    threshold --- Minimum confidence score.
    top_k --- Number of strongest matches to keep.

    Returns
    -------
    tuple --- Filtered keypoints and confidence scores.
    """

    # Keep only correspondences above the confidence threshold
    mask = confidence > threshold

    keypoints0 = keypoints0[mask]
    keypoints1 = keypoints1[mask]
    confidence = confidence[mask]

    # Sort correspondences by confidence in descending order
    order = confidence.argsort()[::-1]

    # Retain only the strongest correspondences
    keypoints0 = keypoints0[order][:top_k]
    keypoints1 = keypoints1[order][:top_k]
    confidence = confidence[order][:top_k]

    return keypoints0, keypoints1, confidence


def visualize_matches(imgA, imgB, keypoints0, keypoints1, confidence):
    """
    Visualize matched keypoints between two images.

    Parameters
    ----------
    imgA --- First RGB image.
    imgB --- Second RGB image.
    keypoints0 --- Matched keypoints in the first image.
    keypoints1 --- Matched keypoints in the second image.
    confidence --- Confidence score for every correspondence.
    """

    h1, w1 = imgA.shape[:2]
    h2, w2 = imgB.shape[:2]

    # Create a canvas large enough to display both images side by side
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    canvas[:h1, :w1] = imgA
    canvas[:h2, w1:] = imgB

    plt.figure(figsize=(18, 10))
    plt.imshow(canvas)

    # Draw correspondences using confidence-based colors
    for p1, p2, c in zip(keypoints0, keypoints1, confidence):

        color = plt.cm.viridis(c)

        plt.plot(
            [p1[0], p2[0] + w1],
            [p1[1], p2[1]],
            color=color,
            linewidth=1
        )

        plt.scatter(
            p1[0],
            p1[1],
            color=color,
            s=8
        )

        plt.scatter(
            p2[0] + w1,
            p2[1],
            color=color,
            s=8
        )

    plt.axis("off")

    # Save the visualization and display it
    plt.savefig(OUTPUT_DIR / "matches.png", dpi=300, bbox_inches="tight")
        
    plt.show()


# ==================================================
# Pipeline execution
# ==================================================

imgA, imgB, kp0, kp1, conf = run_matching(imageA_path, imageB_path, matcher, device)

kp0, kp1, conf = filter_matches(kp0, kp1, conf, CONFIDENCE_THRESHOLD, TOP_K = 100)

visualize_matches(imgA, imgB, kp0, kp1, conf)
