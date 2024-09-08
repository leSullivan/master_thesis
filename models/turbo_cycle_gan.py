import torch
import lpips
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from models import Generator, Discriminator  # Bgssuming you have defined these already
from .utils import preprocess_for_fid, DinoStructureLoss


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        norm_type,
        d_type,
        ndf,
        nd_layers,
        g_type,
        ngf,
        n_downsampling,
        lambda_cycle,
        lambda_identity,
        calculate_scores_during_training,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator_Bg2Fence = UNet(
            ngf=ngf,
            n_downsampling=n_downsampling,
            norm_type=norm_type,
            g_type=g_type,
        )
        self.generator_Fence2Bg = Generator(
            ngf=ngf,
            n_downsampling=n_downsampling,
            norm_type=norm_type,
            g_type=g_type,
        )

        self.discriminator_Bg = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
        )
        self.discriminator_Fence = Discriminator(
            ndf=ndf,
            nd_layers=nd_layers,
            norm_type=norm_type,
            d_type=d_type,
        )

        self.criterion_gan = self.init_adv_loss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = lpips.LPIPS(net="vgg").to(self.device)

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.fid = FrechetInceptionDistance(reset_real_features=False)
        self.structure_loss = DinoStructureLoss(device=self.device)

        self.calculate_scores_during_training = calculate_scores_during_training
        self.automatic_optimization = False

    def forward(self, x, direction="Bg2Fence"):
        if direction == "Bg2Fence":
            return self.generator_Bg2Fence(x)
        else:
            return self.generator_Fence2Bg(x)

    def init_adv_loss(self):
        if self.hparams["d_type"] == "basic":
            return nn.BCELoss()
        else:
            return nn.MSELoss()

    def training_step(self, batch, batch_idx):
        bg_imgs, fence_imgs = batch["background"], batch["fence"]
        optimizer_G, optimizer_D = self.optimizers()

        fake_fences = self.generator_Bg2Fence(bg_imgs)
        fake_bgs = self.generator_Fence2Bg(fence_imgs)

        rec_fences = self.generator_Bg2Fence(fake_bgs)
        rec_bgs = self.generator_Fence2Bg(fake_fences)

        # cycle loss
        loss_cycle_bg = self.criterion_cycle(rec_bgs, bg_imgs)
        loss_cycle_fence = self.criterion_cycle(rec_fences, fence_imgs)
        loss_cycle = loss_cycle_bg + loss_cycle_fence

        self.log("loss_cycle", loss_cycle, on_step=True, on_epoch=True)

        # identity loss
        loss_identity_bg = self.criterion_identity(
            self.generator_Fence2Bg(bg_imgs), bg_imgs
        )
        loss_identity_fence = self.criterion_identity(
            self.generator_Bg2Fence(fence_imgs), fence_imgs
        )
        loss_identity = loss_identity_bg + loss_identity_fence

        self.log(
            "loss_identity",
            loss_identity,
            on_step=True,
            on_epoch=True,
        )

        # adversarial loss
        pred_fake_fence = self.discriminator_Fence(fake_fences)
        loss_gan_Bg2Fence = self.criterion_gan(
            pred_fake_fence, torch.ones_like(pred_fake_fence)
        )
        self.log(
            "loss_adv_Bg2Fence",
            loss_gan_Bg2Fence,
            on_step=True,
            on_epoch=True,
        )

        pred_fake_bg = self.discriminator_Bg(fake_bgs)
        loss_gan_Fence2Bg = self.criterion_gan(
            pred_fake_bg, torch.ones_like(pred_fake_bg)
        )
        self.log(
            "loss_adv_Fence2Bg",
            loss_gan_Fence2Bg,
            on_step=True,
            on_epoch=True,
        )

        loss_G = (
            loss_gan_Bg2Fence
            + loss_gan_Fence2Bg
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
        )

        self.log("loss_G", loss_G, prog_bar=True, on_step=True, on_epoch=True)

        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()

        # ----------------------------------------------------------------------------------

        pred_bg_imgs = self.discriminator_Bg(bg_imgs)
        pred_fake_bg = self.discriminator_Bg(fake_bgs.detach())
        loss_D_bg_imgs = self.criterion_gan(pred_bg_imgs, torch.ones_like(pred_bg_imgs))
        loss_D_fake_bg = self.criterion_gan(
            pred_fake_bg, torch.zeros_like(pred_fake_bg)
        )
        loss_D_Bg = (loss_D_bg_imgs + loss_D_fake_bg) / 2

        self.log("loss_D_Bg", loss_D_Bg, on_step=True, on_epoch=True)

        pred_fence_imgs = self.discriminator_Fence(fence_imgs)
        pred_fake_fence = self.discriminator_Fence(fake_fences.detach())
        loss_D_fence_imgs = self.criterion_gan(
            pred_fence_imgs, torch.ones_like(pred_fence_imgs)
        )
        loss_D_fake_fence = self.criterion_gan(
            pred_fake_fence, torch.zeros_like(pred_fake_fence)
        )
        loss_D_Fence = (loss_D_fence_imgs + loss_D_fake_fence) / 2

        self.log("loss_D_Fence", loss_D_Fence, on_step=True, on_epoch=True)

        loss_D = loss_D_Bg + loss_D_Fence

        self.log("loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=True)

        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()

        if not self.calculate_scores_during_training:
            return

        if self.current_epoch % 20 == 0 and self.current_epoch != 0:
            norm_fake_fence = preprocess_for_fid(fake_fences)
            self.fid.update(norm_fake_fence, real=False)
            self.structure_loss.update_dino_struct_loss(bg_imgs, fake_fences)

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
            val_dataloader = self.trainer.datamodule.val_dataloader()
            loader_Bg = val_dataloader["background"]
            loader_Fence = val_dataloader["fence"]

            for batch_Bg, batch_Fence in zip(loader_Bg, loader_Fence):
                bg_imgs = batch_Bg.to(self.device)
                fence_imgs = batch_Fence.to(self.device)

                fake_fence = self.generator_Bg2Fence(bg_imgs)
                fake_bg = self.generator_Fence2Bg(fence_imgs)

                grid = make_grid(
                    torch.cat((bg_imgs, fake_fence, fence_imgs, fake_bg), dim=0),
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
            list(self.generator_Bg2Fence.parameters())
            + list(self.generator_Fence2Bg.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        optimizer_D = torch.optim.AdamW(
            list(self.discriminator_Bg.parameters())
            + list(self.discriminator_Fence.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [optimizer_G, optimizer_D], []
