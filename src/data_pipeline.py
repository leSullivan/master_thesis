import os
import pytorch_lightning as pl

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import IMG_CH, TRAIN_IMG_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnpairedImageDataModule(pl.LightningDataModule):
    def __init__(self, img_h, img_w, batch_size, crop, num_workers=4):
        super().__init__()
        self.data_dir = TRAIN_IMG_PATH
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop = crop
        self.img_h = img_h
        self.img_w = img_w

        # Conditional transform based on `crop` flag
        self.transform = transforms.Compose(
            [
                (
                    transforms.CenterCrop((self.img_h, self.img_w))
                    if self.crop
                    else transforms.Resize((self.img_h, self.img_w))
                ),
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
            self.val_bg = UnpairedImageDataset(self.data_dir, "valBg", self.transform)
            self.val_fence = UnpairedImageDataset(
                self.data_dir, "valFence", self.transform
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

    def val_dataloader(self):
        val_dataset = PairedImageDataset(self.val_bg, self.val_fence)
        return DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers,
        )


class PairedImageDataset(Dataset):
    def __init__(self, bg_dataset, fence_dataset):
        self.bg_dataset = bg_dataset
        self.fence_dataset = fence_dataset
        assert len(self.bg_dataset) == len(
            self.fence_dataset
        ), "Datasets must have the same length"

    def __len__(self):
        return len(self.bg_dataset)

    def __getitem__(self, idx):
        bg_image = self.bg_dataset[idx]
        fence_image = self.fence_dataset[idx]
        return bg_image, fence_image


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
