from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithm import (
    read_rgb,
    run_algorithm
)

def filter_matches(kp0, kp1, confidence, threshold=0.8):
    """
    Remove low-confidence matches.
    """

    mask = confidence > threshold

    return (
        kp0[mask],
        kp1[mask],
        confidence[mask]
    )

def sort_matches(kp0, kp1, confidence):

    idx = confidence.argsort()[::-1]

    return (
        kp0[idx],
        kp1[idx],
        confidence[idx]
    )

def keep_best_matches(kp0, kp1, confidence, top_k=100):

    return (
        kp0[:top_k],
        kp1[:top_k],
        confidence[:top_k]
    )

def visualize_matches(
    img1,
    img2,
    pts1,
    pts2,
    confidence,
    save_path=None
):
    """
    Visualize matched keypoints between two images.

    Parameters
    ----------
    img1 : ndarray
    img2 : ndarray
    pts1 : ndarray
    pts2 : ndarray
    confidence : ndarray
    save_path : Path or None
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros(
        (max(h1, h2), w1 + w2, 3),
        dtype=np.uint8
    )

    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    plt.figure(figsize=(18, 10))
    plt.imshow(canvas)

    for p1, p2, c in zip(pts1, pts2, confidence):

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

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
