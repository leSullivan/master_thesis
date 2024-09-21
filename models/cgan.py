import torch
import lpips
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

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
        d_use_sigmoid,
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
        )

        self.discriminator = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
            d_use_sigmoid=d_use_sigmoid,
            device=self.device,
        )

        self.criterion_gan = init_gan_loss(d_type=d_type)
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = (
            lpips.LPIPS(net="vgg").to(self.device).requires_grad_(False)
        )

        self.fid = FrechetInceptionDistance()
        self.structure_loss = DinoStructureLoss(device=self.device)

        self.calculate_scores_during_training = calculate_scores_during_training

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        bg_imgs, fence_imgs = batch["background"], batch["fence"]

        optimizer_G, optimizer_D = self.optimizers()

        generated_fences = self.generator(bg_imgs)

        grid = make_grid(
            torch.cat((generated_fences, fence_imgs), dim=0),
            nrow=4,
            normalize=True,
        )

        self.logger.experiment.add_image(
            "Generated_Images", grid, self.current_epoch
        )

        if self.hparams["d_type"] == "vagan":
            pred_fake = self.discriminator(generated_fences, for_G=True)
            loss_gan = pred_fake.mean()
        else:
            pred_fake = self.discriminator(generated_fences)
            loss_gan = self.criterion_gan(
                pred_fake, torch.ones_like(pred_fake, device=self.device)
            )

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

        if self.hparams["d_type"] == "vagan":
            pred_real = self.discriminator(fence_imgs)
            pred_fake = self.discriminator(generated_fences.detach())
            loss_real = pred_real.mean()
            loss_fake = pred_fake.mean()
        else:
            pred_real = self.discriminator(fence_imgs, for_real=True)
            pred_fake = self.discriminator(generated_fences.detach(), for_real=False)
            loss_real = self.criterion_gan(
                pred_real, torch.ones_like(pred_real, device=self.device)
            )
            loss_fake = self.criterion_gan(
                pred_fake, torch.zeros_like(pred_fake, device=self.device)
            )

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

    def on_train_epoch_end(self):
        # if (
        #     not self.current_epoch % 20 == 0
        #     or self.current_epoch == 0
        #     or not self.calculate_scores_during_training
        # ):
        #     return

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

            self.logger.experiment.add_image(
                "Generated_Images", grid, self.current_epoch
            )

            # norm_fence_imgs = preprocess_for_fid(fence_imgs)
            # norm_fake_fences = preprocess_for_fid(fake_fence_imgs)
            # self.fid.update(norm_fence_imgs, real=True)
            # self.fid.update(norm_fake_fences, real=False)
            # fid_score = self.fid.compute().item()
            # self.log("FID", fid_score, on_epoch=True)
            # self.fid.reset()

            # self.structure_loss.update_dino_struct_loss(bg_imgs, fake_fence_imgs)
            # structure_loss = self.structure_loss.compute()
            # self.log("DINO", structure_loss, on_epoch=True)
            # self.structure_loss.reset()

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
