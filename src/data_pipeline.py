import os
import pytorch_lightning as pl

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import IMG_CH, TRAIN_IMG_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnpairedImageDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = os.path.join(root_dir, domain)
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(self.root_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class UnpairedImageDataModule(pl.LightningDataModule):
    def __init__(self, img_h, img_w, batch_size, num_workers=4):
        super().__init__()
        self.data_dir = TRAIN_IMG_PATH
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * IMG_CH, (0.5,) * IMG_CH),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_bg = UnpairedImageDataset(
                self.data_dir, "trainBg", self.transform
            )
            self.train_fence = UnpairedImageDataset(
                self.data_dir, "trainFence", self.transform
            )

    def train_dataloader(self):
        loader_A = DataLoader(
            self.train_bg,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        loader_B = DataLoader(
            self.train_fence,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return {"background": loader_A, "fence": loader_B}
