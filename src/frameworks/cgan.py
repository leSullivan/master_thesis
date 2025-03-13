import torch
import lpips
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from src.models import Generator, Discriminator
from src.utils import preprocess_for_fid, init_gan_loss, DinoStructureLoss


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
        )

        self.discriminator = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
            d_use_sigmoid=d_use_sigmoid,
        )

        self.criterion_gan = init_gan_loss(d_type=d_type)
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = lpips.LPIPS(net="vgg").requires_grad_(False)

        self.fid = FrechetInceptionDistance(reset_real_features=False)
        self.fake_imgs = []
        self.structure_loss = DinoStructureLoss()

        self.calculate_scores_during_training = calculate_scores_during_training

        self.automatic_optimization = False

    def on_train_start(self):
        imgs = [
            fence_imgs
            for fence_imgs in self.trainer.datamodule.train_dataloader()["fence"]
        ]

        fence_imgs = torch.cat([*imgs], dim=0).to(self.device)
        fence_imgs = preprocess_for_fid(fence_imgs)
        self.fid.update(fence_imgs, real=True)

    def training_step(self, batch, batch_idx):
        bg_imgs, fence_imgs = batch["background"], batch["fence"]

        optimizer_G, optimizer_D = self.optimizers()

        generated_fences = self.generator(bg_imgs)

        if self.hparams["d_type"] == "vagan":
            pred_fake = self.discriminator(generated_fences, for_G=True)
            loss_gan = pred_fake.mean()
        else:
            pred_fake = self.discriminator(generated_fences)
            loss_gan = self.criterion_gan(
                pred_fake, torch.ones_like(pred_fake, device=self.device)
            )

        self.log("generator/loss_adv", loss_gan, on_step=True, on_epoch=True)

        loss_G = loss_gan * self.hparams["lambda_gan"]

        if self.hparams["lambda_identity"] > 0:
            loss_l1 = self.criterion_identity(bg_imgs, generated_fences)
            self.log("generator/loss_identity", loss_l1, on_step=True, on_epoch=True)
            loss_G += self.hparams["lambda_identity"] * loss_l1

        if self.hparams["lambda_perceptual"] > 0:
            loss_perception = self.criterion_perceptual(
                fence_imgs, generated_fences
            ).mean()
            self.log("generator/loss_perceptual", loss_l1, on_step=True, on_epoch=True)
            loss_G += self.hparams["lambda_perceptual"] * loss_perception

        self.log(
            "generator/loss_G",
            loss_G,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()

        if self.hparams["d_type"] == "vagan":
            pred_real = self.discriminator(fence_imgs, for_real=True)
            pred_fake = self.discriminator(generated_fences.detach(), for_real=False)
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

        self.log("discriminator/loss_D_real", loss_real, on_step=True, on_epoch=True)
        self.log("discriminator/loss_D_fake", loss_fake, on_step=True, on_epoch=True)

        loss_D = (loss_real + loss_fake) * self.hparams["lambda_gan"]

        self.log(
            "discriminator/loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=True
        )

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()

        if self.current_epoch % 20 == 0 and self.current_epoch != 0:
            self.fake_imgs.append(generated_fences)

    def validation_step(self, batch, batch_idx):
        if (
            not self.current_epoch % 20 == 0
            or self.current_epoch == 0
            or not self.calculate_scores_during_training
        ):
            return

        bg_imgs, fence_imgs = batch

        generated_fences = self.generator(bg_imgs)

        grid = make_grid(
            torch.cat((bg_imgs, generated_fences), dim=0),
            nrow=4,
            normalize=True,
        )

        self.logger.experiment.add_image(
            "eval/Generated_Images", grid, self.current_epoch
        )

        fake_fence_imgs = torch.cat([*self.fake_imgs], dim=0).to(self.device)
        norm_gen_fences = preprocess_for_fid(fake_fence_imgs)
        self.fid.update(norm_gen_fences, real=False)

        self.structure_loss.update_dino_struct_loss(bg_imgs, generated_fences)

    def on_validation_epoch_end(self):
        if (
            not self.current_epoch % 20 == 0
            or self.current_epoch == 0
            or not self.calculate_scores_during_training
        ):
            return
        fid_score = self.fid.compute().item()
        self.log("eval/FID", fid_score, on_epoch=True)
        self.fid.reset()
        self.fake_imgs = []

        structure_loss = self.structure_loss.compute()
        self.log("eval/DINO", structure_loss)
        self.structure_loss.reset()

    def on_epoch_end(self):
        scheduler_G, scheduler_D = self.lr_schedulers()
        scheduler_G.step()
        scheduler_D.step()

    def configure_optimizers(self):
        optimizer_G = torch.optim.AdamW(
            list(self.generator.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        optimizer_D = torch.optim.AdamW(
            list(self.discriminator.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        scheduler_G = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer_G, step_size=80, gamma=0.5
            ),
            "name": "lr/_optimizer_G",
            "interval": "epoch",
            "frequency": 1,
        }
        scheduler_D = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer_D, step_size=80, gamma=0.5
            ),
            "name": "lr/_optimizer_D",
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]
