import os

import cv2
import numpy as np
from tqdm import tqdm

root = "data"

img_folder = os.path.join(root, "CelebA-HQ-img")
mask_folder = os.path.join(root, "CelebAMask-HQ-mask-anno")
dest_folder = os.path.join(root, "CelebA-HQ-img-masked")
os.makedirs(dest_folder, exist_ok=True)
parts = [
    "hair",
    "l_brow",
    "l_eye",
    "l_ear",
    "l_lip",
    "mouth",
    "nose",
    "r_brow",
    "r_ear",
    "r_eye",
    "skin",
    "u_lip",
]

for img_file in tqdm(os.listdir(img_folder)):
    name = int(os.path.splitext(img_file)[0])
    img_path = os.path.join(img_folder, img_file)

    mask_folder_path = os.path.join(mask_folder, str(name // 2000))

    masks = [
        cv2.imread(os.path.join(mask_folder_path, f"{name:05d}_{p}.png"))
        for p in parts
        if os.path.exists(os.path.join(mask_folder_path, f"{name:05d}_{p}.png"))
    ]
    masks = np.stack(masks)
    mask = masks.sum(axis=0)
    mask = np.where(mask >= 255, 255, 0).astype(np.uint8)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(img_path)

    mask = cv2.resize(mask, img.shape[:2])

    masked_img = np.where(mask == 255, img, 255)
    cv2.imwrite(os.path.join(dest_folder, img_file), masked_img)
