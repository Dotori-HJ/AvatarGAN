import argparse
import glob
import os
from multiprocessing import Pool

from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/cartoonset100k")
    parser.add_argument("--dest", type=str, default="data/cartoon")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def work(i):
    for _ in range(3):  # try three times
        try:
            img_path = img_paths[i]
            img_name = os.path.split(img_path)[-1]
            dest_path = os.path.join(dest, img_name)

            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            img.save(dest_path)
            succeed = True
            break
        except:
            succeed = False
    if not succeed:
        print("%s fails!" % img_paths[i])


if __name__ == "__main__":
    args = get_args()
    root = args.root
    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    transform = transforms.Compose([transforms.CenterCrop(350), transforms.Resize((128, 128))])

    img_paths = []
    for folder in os.listdir(root):
        img_paths += glob.glob(os.path.join(root, folder, "*.png"))

    pool = Pool(args.num_workers)
    for i in tqdm(pool.imap(work, range(len(img_paths))), total=len(img_paths)):
        pass
    pool.close()
    pool.join()
