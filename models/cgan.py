import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from models import Generator, Discriminator
from .utils import normalize_tensor


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
        *args,
        **kwargs,
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

        self.criterion_gan = self.init_adv_loss()
        self.criterion_identity = nn.L1Loss()

        self.fid = FrechetInceptionDistance(
            feature=2048, normalize=True, reset_real_features=False
        )

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def init_adv_loss(self):
        if self.hparams["discriminator_type"] == "basic":
            return nn.BCELoss()
        else:
            return nn.MSELoss()

    def on_train_start(self):
        for fence_imgs in self.trainer.datamodule.train_dataloader()["fence"]:
            fence_imgs = fence_imgs.to(self.device)
            fence_imgs = normalize_tensor(fence_imgs)
            self.fid.update(fence_imgs, real=True)

    def training_step(self, batch, batch_idx):
        bg_img, fence_imgs = batch["background"], batch["fence"]

        optimizer_G, optimizer_D = self.optimizers()

        generated_fences = self.generator(bg_img)

        pred_fake = self.discriminator(generated_fences)

        loss_adv = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        loss_L1 = self.criterion_identity(generated_fences, bg_img)

        loss_G = loss_adv + self.hparams.lamba_identity * loss_L1

        self.log(
            "train/_loss_adv", loss_adv, prog_bar=True, on_step=True, on_epoch=True
        )
        self.log("train/_loss_L1", loss_L1, prog_bar=True, on_step=True, on_epoch=True)
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
        loss_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))

        pred_fake = self.discriminator(generated_fences.detach())
        loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_real + loss_fake) / 2

        self.log("train/loss_D_real", loss_real, on_step=True, on_epoch=True)
        self.log("train/loss_D_fake", loss_fake, on_step=True, on_epoch=True)
        self.log("train/loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=True)

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()

        if self.current_epoch % 10 == 0 and self.current_epoch != 0:
            norm_gen_fences = normalize_tensor(generated_fences)
            self.fid.update(norm_gen_fences.to(torch.float32), real=False)

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
            val_dataloader = self.trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            bg_imgs, fence_imgs = batch["background"], batch["fence"]

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
