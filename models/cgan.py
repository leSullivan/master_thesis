import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from models import Generator, Discriminator


class CGAN(pl.LightningModule):
    def __init__(
        self,
        lr,
        beta1,
        beta2,
        img_h,
        img_w,
        lamba_identity,
        norm_type,
        discriminator_type,
        ndf,
        nd_layers,
        generator_type,
        ngf,
        n_downsampling,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            ngf=ngf,
            n_downsampling=n_downsampling,
            norm_type=norm_type,
            generator_type=generator_type,
        )
        self.discriminator = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            discriminator_type=discriminator_type,
        )

        self.criterion_gan = nn.MSELoss()

        self.criterion_identity = nn.L1Loss()

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

        self.generated_fences = self.generator(bg_img)

        pred_fake = self.discriminator(self.generated_fences)

        loss_GAN = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        loss_L1 = self.criterion_identity(self.generated_fences, bg_img)

        loss_G = loss_GAN + self.hparams.lamba_identity * loss_L1

        self.log("train/_loss_G", loss_G, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/_loss_L1", loss_L1, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/_loss_G",
            loss_G,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("train/_loss_L1", loss_L1, prog_bar=True, on_step=True, on_epoch=True)

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

        if self.current_epoch % 10 == 0 and self.current_epoch != 0:
            self.fid.update(fence_imgs.to(torch.float32), real=True)
            self.fid.update(self.generated_fences.to(torch.float32), real=False)

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
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

        if self.current_epoch % 10 == 0 and self.current_epoch != 0:

            real_images = torch.cat(self.fid_real_features)
            fake_images = torch.cat(self.fid_fake_features)

            self.fid.update(real_images, real=True)
            self.fid.update(fake_images, real=False)

            fid_score = self.fid.compute().item()
            self.log("train/FID", fid_score, prog_bar=True)

            self.fid.reset()

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
