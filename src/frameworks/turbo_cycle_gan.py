import torch
import lpips
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from src.models import SDTurboGenerator, Discriminator
from src.utils import preprocess_for_fid, init_weights, init_gan_loss, DinoStructureLoss


class TurboCycleGAN(pl.LightningModule):
    def __init__(
        self,
        norm_type,
        d_type,
        ndf,
        nd_layers,
        prompt_bg,
        prompt_fence,
        d_use_sigmoid,
        calculate_scores_during_training,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = SDTurboGenerator(
            prompt_bg=prompt_bg,
            prompt_fence=prompt_fence,
        )

        self.discriminator_Bg = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
            d_use_sigmoid=d_use_sigmoid,
        )

        self.discriminator_Fence = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
            d_use_sigmoid=d_use_sigmoid,
        )

        self.fid = FrechetInceptionDistance()
        self.structure_loss = DinoStructureLoss()

        init_weights(self.discriminator_Bg, d_type)
        init_weights(self.discriminator_Fence, d_type)

        self.criterion_gan = init_gan_loss(d_type=d_type)
        self.criterion_cycle = nn.L1Loss()
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
        optimizer_G, optimizer_D = self.optimizers()
        bg_imgs, fence_imgs = batch["background"], batch["fence"]

        fake_fences = self.generator.forward(bg_imgs, "Bg2Fence")
        fake_bgs = self.generator.forward(fence_imgs, "Fence2Bg")

        rec_fences = self.generator.forward(fake_bgs, "Bg2Fence")
        rec_bgs = self.generator.forward(fake_fences, "Fence2Bg")

        # Cycle loss
        loss_cycle_bg = self.criterion_cycle(rec_bgs, bg_imgs)
        loss_cycle_fence = self.criterion_cycle(rec_fences, fence_imgs)
        loss_cycle = loss_cycle_bg + loss_cycle_fence
        self.log("generator/loss_cycle", loss_cycle)

        # Perceptual loss
        loss_perceptual_fence = self.criterion_perceptual(
            fence_imgs, fake_fences
        ).mean()
        loss_perceptual_bg = self.criterion_perceptual(bg_imgs, fake_bgs).mean()
        loss_perceptual = loss_perceptual_bg + loss_perceptual_fence
        self.log("generator/loss_perceptual", loss_perceptual)

        # Adversarial loss
        if self.hparams["d_type"] == "vagan":
            pred_fake = self.discriminator_Fence(fake_fences, for_G=True)
            loss_gan_Bg2Fence = pred_fake.mean()
        else:
            pred_fake = self.discriminator_Fence(fake_fences)
            loss_gan_Bg2Fence = self.criterion_gan(
                pred_fake, torch.ones_like(pred_fake, device=self.device)
            )
        self.log("generator/loss_adv_Bg2Fence", loss_gan_Bg2Fence, on_epoch=True)

        if self.hparams["d_type"] == "vagan":
            pred_fake = self.discriminator_Bg(fake_bgs, for_G=True)
            loss_gan_Fence2Bg = pred_fake.mean()
        else:
            pred_fake = self.discriminator_Bg(fake_bgs)
            loss_gan_Fence2Bg = self.criterion_gan(
                pred_fake, torch.ones_like(pred_fake, device=self.device)
            )
        self.log("generator/loss_adv_Fence2Bg", loss_gan_Fence2Bg, on_epoch=True)

        loss_adv = loss_gan_Bg2Fence + loss_gan_Fence2Bg
        self.log("generator/loss_adv", loss_adv, on_epoch=True)

        # Total generator loss
        loss_G = (
            loss_adv * self.hparams["lambda_gan"]
            + self.hparams["lambda_cycle"] * loss_cycle
            + self.hparams["lambda_perceptual"] * loss_perceptual
        )
        self.log("generator/loss_G", loss_G, prog_bar=True, on_epoch=True)

        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()

        # ----------------------------------------------------------------------------------

        if self.hparams["d_type"] == "vagan":
            pred_real = self.discriminator_Bg(bg_imgs, for_real=True)
            pred_fake = self.discriminator_Bg(fake_bgs.detach(), for_real=False)
            loss_D_bg_imgs = pred_real.mean()
            loss_D_fake_bg = pred_fake.mean()
        else:
            pred_real = self.discriminator_Bg(bg_imgs, for_real=True)
            pred_fake = self.discriminator_Bg(fake_bgs.detach(), for_real=False)
            loss_D_bg_imgs = self.criterion_gan(
                pred_real, torch.ones_like(pred_real, device=self.device)
            )
            loss_D_fake_bg = self.criterion_gan(
                pred_fake, torch.zeros_like(pred_fake, device=self.device)
            )

        loss_D_Bg = (loss_D_bg_imgs + loss_D_fake_bg) * self.hparams["lambda_gan"]

        self.log("discriminator/loss_D_Bg", loss_D_Bg, on_epoch=True)

        if self.hparams["d_type"] == "vagan":
            pred_real = self.discriminator_Fence(fence_imgs, for_real=True)
            pred_fake = self.discriminator_Fence(fake_fences.detach(), for_real=False)
            loss_D_fence_imgs = pred_real.mean()
            loss_D_fake_fence = pred_fake.mean()
        else:
            pred_real = self.discriminator_Fence(fence_imgs, for_real=True)
            pred_fake = self.discriminator_Fence(fake_fences.detach(), for_real=False)
            loss_D_fence_imgs = self.criterion_gan(
                pred_real, torch.ones_like(pred_real, device=self.device)
            )
            loss_D_fake_fence = self.criterion_gan(
                pred_fake, torch.zeros_like(pred_fake, device=self.device)
            )

        loss_D_Fence = (loss_D_fence_imgs + loss_D_fake_fence) * self.hparams[
            "lambda_gan"
        ]

        self.log("discriminator/loss_D_Fence", loss_D_Fence, on_epoch=True)

        loss_D = loss_D_Bg + loss_D_Fence

        self.log("discriminator/loss_D", loss_D, prog_bar=True, on_epoch=True)

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()

        if self.current_epoch % 20 == 0 and self.current_epoch != 0:
            self.fake_imgs.append(fake_fences)

    def validation_step(self, batch, batch_idx):
        if (
            not self.current_epoch % 20 == 0
            or self.current_epoch == 0
            or not self.calculate_scores_during_training
        ):
            return

        bg_imgs, _ = batch

        fake_fence_imgs = self.generator.forward(bg_imgs, "Bg2Fence")

        grid = make_grid(
            torch.cat((bg_imgs, fake_fence_imgs), dim=0),
            nrow=4,
            normalize=True,
        )

        self.logger.experiment.add_image("Generated_Images", grid, self.current_epoch)

        all_fake_fence_imgs = torch.cat([*self.fake_imgs], dim=0).to(bg_imgs.device)
        norm_gen_fences = preprocess_for_fid(all_fake_fence_imgs)
        self.fid.update(norm_gen_fences, real=False)

        self.structure_loss.update_dino_struct_loss(bg_imgs, fake_fence_imgs)

    def on_validation_epoch_end(self):
        if not self.current_epoch % 20 == 0 or self.current_epoch == 0:
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
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps,
        )
        optimizer_D = torch.optim.AdamW(
            list(self.discriminator_Bg.parameters())
            + list(self.discriminator_Fence.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps,
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
