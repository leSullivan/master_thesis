import torch
import lpips
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

import lpips.weights
from models import Generator, Discriminator
from .utils import preprocess_for_fid, init_gan_loss, DinoStructureLoss


class CGAN(pl.LightningModule):
    def __init__(
        self,
        norm_type,
        d_type,
        ndf,
        nd_layers,
        g_type,
        ngf,
        n_downsampling,
        calculate_scores_during_training,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(
            ngf=ngf,
            n_downsampling=n_downsampling,
            norm_type=norm_type,
            g_type=g_type,
            device=self.device,
        ).to(self.device)

        self.discriminator = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
            d_use_sigmoid=False,
            device=self.device,
        ).to(self.device)

        self.criterion_gan = init_gan_loss(d_type=d_type)
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = (
            lpips.LPIPS(net="vgg").to(self.device).requires_grad_(False)
        )

        self.fid = FrechetInceptionDistance(reset_real_features=False)
        self.structure_loss = DinoStructureLoss(device=self.device)

        self.calculate_scores_during_training = calculate_scores_during_training

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def on_train_start(self):
        if not self.calculate_scores_during_training:
            return
        for fence_imgs in self.trainer.datamodule.train_dataloader()["fence"]:
            fence_imgs = fence_imgs.to(self.device)
            fence_imgs = preprocess_for_fid(fence_imgs)
            self.fid.update(fence_imgs, real=True)

    def training_step(self, batch, batch_idx):
        bg_imgs, fence_imgs = batch["background"], batch["fence"]
        print(f"Background images device: {bg_imgs.device}")
        print(f"Fence images device: {fence_imgs.device}")

        optimizer_G, optimizer_D = self.optimizers()

        generated_fences = self.generator(bg_imgs)
        print(f"Generated fences device: {generated_fences.device}")

        pred_fake = self.discriminator(generated_fences)

        if self.hparams["d_type"] == "vagan":
            loss_gan = pred_fake.mean()
        else:
            loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        self.log("loss_adv", loss_gan, on_step=True, on_epoch=True)

        loss_G = loss_gan * self.hparams["lambda_gan"]

        if self.hparams["lambda_identity"] > 0:
            loss_l1 = self.criterion_identity(bg_imgs, generated_fences)
            self.log("loss_identity", loss_l1, on_step=True, on_epoch=True)
            loss_G += self.hparams["lambda_identity"] * loss_l1

        if self.hparams["lambda_perceptual"] > 0:
            loss_perception = self.criterion_perceptual(
                fence_imgs, generated_fences
            ).mean()
            self.log("loss_perceptual", loss_l1, on_step=True, on_epoch=True)
            loss_G += self.hparams["lambda_perceptual"] * loss_perception

        self.log(
            "loss_G",
            loss_G,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()

        pred_real = self.discriminator(fence_imgs)
        pred_fake = self.discriminator(generated_fences.detach())

        if self.hparams["d_type"] == "vagan":
            loss_real = pred_real.mean()
            loss_fake = pred_fake.mean()
        else:
            loss_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))

        self.log("loss_D_real", loss_real, on_step=True, on_epoch=True)
        self.log("loss_D_fake", loss_fake, on_step=True, on_epoch=True)

        loss_D = (
            loss_real * self.hparams["lambda_gan"]
            + loss_fake * self.hparams["lambda_gan"]
        ) / 2

        self.log("loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=True)

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()

        if not self.calculate_scores_during_training:
            return

        if self.current_epoch % 20 == 0 and self.current_epoch != 0:
            norm_gen_fences = preprocess_for_fid(generated_fences)
            self.fid.update(norm_gen_fences, real=False)
            self.structure_loss.update_dino_struct_loss(bg_imgs, generated_fences)

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
            val_dataloader = self.trainer.datamodule.val_dataloader()
            loader_A = val_dataloader["background"]
            loader_B = val_dataloader["fence"]

            for batch_A, batch_B in zip(loader_A, loader_B):
                bg_imgs = batch_A.to(self.device)
                fence_imgs = batch_B.to(self.device)

                fake_fence_imgs = self.generator(bg_imgs)

                grid = make_grid(
                    torch.cat((bg_imgs, fake_fence_imgs, fence_imgs), dim=0),
                    nrow=4,
                    normalize=True,
                )

            fake_fence_imgs = self.generator(bg_imgs)

            grid = make_grid(
                torch.cat((bg_imgs, fake_fence_imgs, fence_imgs), dim=0),
                nrow=4,
                normalize=True,
            )

            self.logger.experiment.add_image(
                "Generated_Images", grid, self.current_epoch
            )

        if (
            self.current_epoch % 20 == 0
            and self.current_epoch != 0
            and self.calculate_scores_during_training
        ):
            fid_score = self.fid.compute().item()
            self.log("FID", fid_score, on_epoch=True)
            self.fid.reset()

            structure_loss = self.structure_loss.compute()
            self.log("DINO", structure_loss, on_epoch=True)
            self.structure_loss.reset()

    def configure_optimizers(self):
        optimizer_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        optimizer_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [optimizer_G, optimizer_D], []
