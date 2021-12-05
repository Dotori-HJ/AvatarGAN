import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from configs import get_args
from dataset import Dataset
from models import Decoder, Encoder

IMG_EXTENSIONS = (".jpg", ".png", ".jpeg")


class Dataset:
    def __init__(self, root, transform=None):
        self.root = root
        self.img_paths = [
            os.path.join(root, img_name)
            for img_name in os.listdir(root)
            if img_name.lower().endswith(IMG_EXTENSIONS)
        ]

        self.transform = transform

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i])

        if self.transform is not None:
            img = self.transform(img)

        return img, os.path.split(self.img_paths[i])[-1]

    def __len__(self):
        return len(self.img_paths)


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    src_dataset = Dataset(args.src_root, transform=transform)
    dst_dataset = Dataset(args.dst_root, transform=transform)

    src_loader = DataLoader(
        src_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    dst_loader = DataLoader(
        dst_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    encoder = Encoder(args.num_channels, args.num_features).to(device)
    decoder = Decoder(args.num_channels, args.num_features).to(device)

    exp_dir_path = os.path.join(args.exp_dir, args.exp_name)
    dst_dir_path = os.path.join(args.exp_dir, args.exp_name, "results")
    os.makedirs(dst_dir_path, exist_ok=True)
    os.makedirs(os.path.join(dst_dir_path, "gt"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir_path, "pred"), exist_ok=True)

    ckpt = torch.load(os.path.join(exp_dir_path, "save", "final.ckpt"))
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        pbar = tqdm(zip(src_loader, dst_loader))
        for (src_data, src_name), (dst_data, dst_name) in pbar:
            src_data = src_data.to(device)

            latent = encoder(src_data, mode="AB")
            pred_img = decoder(latent)

            save_image(
                (pred_img + 1.0) * 0.5, os.path.join(dst_dir_path, "pred", src_name[0]), padding=0,
            )
            save_image(
                (dst_data + 1.0) * 0.5, os.path.join(dst_dir_path, "gt", dst_name[0]), padding=0,
            )


if __name__ == "__main__":
    main()
