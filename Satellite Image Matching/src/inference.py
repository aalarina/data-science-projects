import matplotlib.pyplot as plt
import numpy as np

def visualize_matches(img1, img2, pts1, pts2, conf):

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

    for p1, p2, c in zip(pts1, pts2, conf):

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
    plt.show()

mkpts0, mkpts1, confidence = match_all_tiles(
    tilesA,
    tilesB,
    matcher,
    device
)

mask = confidence > 0.8

mkpts0 = mkpts0[mask]
mkpts1 = mkpts1[mask]
confidence = confidence[mask]

print(len(confidence))
print(confidence.mean())
print(confidence.max())

idx = confidence.argsort()[::-1]

mkpts0 = mkpts0[idx]
mkpts1 = mkpts1[idx]
confidence = confidence[idx]

mkpts0 = mkpts0[:100]
mkpts1 = mkpts1[:100]

visualize_matches(
    imgA,
    imgB,
    mkpts0,
    mkpts1,
    confidence
)

