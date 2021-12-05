import os

from PIL import Image

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

        return img

    def __len__(self):
        return len(self.img_paths)
