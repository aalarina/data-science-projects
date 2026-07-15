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


