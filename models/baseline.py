import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid


from src.config import IMG_H, IMG_W, IMG_CH, NOISE_DIM, BETA1, BETA2, LR


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim=NOISE_DIM,
        input_channels=IMG_CH,
        output_channels=IMG_CH,
        image_height=IMG_H,
        image_width=IMG_W,
    ):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.image_height = image_height
        self.image_width = image_width

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + input_channels * image_height * image_width, 512),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU(True)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(2048, output_channels * image_height * image_width), nn.Tanh()
        )

    def forward(self, noise, image):
        image_flat = image.view(image.size(0), -1)
        noise_flat = noise.view(noise.size(0), -1)

        input_combined = torch.cat((noise_flat, image_flat), dim=1)

        x = self.fc1(input_combined)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        x = x.view(x.size(0), self.output_channels, self.image_height, self.image_width)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


class cGAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        lr=LR,
        beta1=BETA1,
        beta2=BETA2,
        noise_dim=NOISE_DIM,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])

        self.generator = generator
        self.discriminator = discriminator

        self.criterion_gan = nn.MSELoss()

        # FID metric
        self.fid = FrechetInceptionDistance(feature=2048)
        self.fid_real_features = []
        self.fid_fake_features = []

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return self.criterion_gan(y_hat, y)

    def training_step(self, batch, batch_idx):
        bg_img, fence_imgs = batch["background"], batch["fence"]

        optimizer_G, optimizer_D = self.optimizers()

        noise = torch.randn(
            bg_img.size(0), self.hparams.noise_dim, 1, 1, device=self.device
        )

        self.generated_fences = self.generator(noise, bg_img)

        pred_fake = self.discriminator(self.generated_fences)
        loss_GAN = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))

        loss_G = loss_GAN

        self.log(
            "train/_loss_G",
            loss_G,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()

        pred_real = self.discriminator(fence_imgs)
        loss_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))

        pred_fake = self.discriminator(self.generated_fences.detach())
        loss_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_real + loss_fake) / 2

        self.log("train/loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_D_real", loss_real, on_step=True, on_epoch=True)
        self.log("train/loss_D_fake", loss_fake, on_step=True, on_epoch=True)

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 & self.current_epoch != 0:
            bg_imgs = next(
                iter(self.trainer.datamodule.train_dataloader()["background"])
            )[:4]
            fence_imgs = next(
                iter(self.trainer.datamodule.train_dataloader()["fence"])
            )[:4]
            fake_fence_imgs = self.generator(bg_imgs)

            grid = make_grid(
                torch.cat((bg_imgs, fake_fence_imgs, fence_imgs), dim=0),
                nrow=4,
                normalize=True,
            )

            self.logger.experiment.add_image(
                "Generated_Images", grid, self.current_epoch
            )

        if self.current_epoch % 10 == 0 & self.current_epoch != 0:

            real_images = torch.cat(self.fid_real_features)
            fake_images = torch.cat(self.fid_fake_features)

            self.fid.update(real_images, real=True)
            self.fid.update(fake_images, real=False)

            fid_score = self.fid.compute().item()
            self.log("train/FID", fid_score, prog_bar=True)

            self.fid_real_features.clear()
            self.fid_fake_features.clear()

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [optimizer_G, optimizer_D], []
