pair = sorted(Path("/prepared_dataset").iterdir())[0]

imgA = read_rgb(pair/"image_A.jp2")
imgB = read_rgb(pair/"image_B.jp2")

plt.figure(figsize=(16,8))

plt.subplot(121)

plt.imshow(imgA)

plt.title("Image A")

plt.subplot(122)

plt.imshow(imgB)

plt.title("Image B")

plt.show()

def split_into_tiles(image, tile_size=1024):
    """
    Split image into non-overlapping tiles.

    Returns:
        tiles - list of dictionaries:
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

            tile = image[
                y:y + tile_size,
                x:x + tile_size
            ]

            tiles.append({
                "image": tile,
                "x": x,
                "y": y
            })

    return tiles

tilesA = split_into_tiles(imgA, tile_size=1024)

tilesB = split_into_tiles(imgB, tile_size=1024)

def match_single_tile(tileA, tileB, matcher, device):

    grayA = cv2.cvtColor(tileA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(tileB, cv2.COLOR_RGB2GRAY)

    tensorA = torch.from_numpy(grayA).float() / 255.
    tensorA = tensorA.unsqueeze(0).unsqueeze(0).to(device)

    tensorB = torch.from_numpy(grayB).float() / 255.
    tensorB = tensorB.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():

        corr = matcher({

            "image0": tensorA,

            "image1": tensorB

        })

    return corr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matcher = KF.LoFTR(pretrained="outdoor").to(device)
matcher.eval()

corr = match_single_tile(

    tilesA[0]["image"],

    tilesB[0]["image"],

    matcher,

    device

)

from tqdm import tqdm

def match_all_tiles(tilesA, tilesB, matcher, device):

    all_keypoints0 = []
    all_keypoints1 = []
    all_confidence = []

    for tileA, tileB in tqdm(
        zip(tilesA, tilesB),
        total=len(tilesA),
        desc="Matching tiles"
    ):

        corr = match_single_tile(
            tileA["image"],
            tileB["image"],
            matcher,
            device
        )

        kp0 = corr["keypoints0"].cpu().numpy()
        kp1 = corr["keypoints1"].cpu().numpy()
        conf = corr["confidence"].cpu().numpy()

        # Переводимо координати у глобальні
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


