import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

from models import Generator, Discriminator


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        lr,
        beta1,
        beta2,
        lambda_cycle,
        lambda_identity,
        norm_type,
        ngf,
        ndf,
        n_downsampling,
        nd_layers,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Generators
        self.G_A2B = Generator(ngf, n_downsampling, norm_type)
        self.G_B2A = Generator(ngf, n_downsampling, norm_type)

        # Discriminators
        self.D_A = Discriminator(ndf, nd_layers, norm_type)
        self.D_B = Discriminator(ndf, nd_layers, norm_type)

        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # FID metric
        self.fid = FrechetInceptionDistance(feature=2048)
        self.fid_real_A = []
        self.fid_fake_A = []
        self.fid_real_B = []
        self.fid_fake_B = []

        self.automatic_optimization = False

    def forward(self, x):
        return self.G_A2B(x)

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch["A"], batch["B"]

        opt_G, opt_D = self.optimizers()

        # Generate fake images
        fake_B = self.G_A2B(real_A)
        fake_A = self.G_B2A(real_B)

        # Train Generators
        # Identity loss
        identity_A = self.G_B2A(real_A)
        identity_B = self.G_A2B(real_B)
        loss_identity_A = self.criterion_identity(identity_A, real_A)
        loss_identity_B = self.criterion_identity(identity_B, real_B)

        # GAN loss
        pred_fake_B = self.D_B(fake_B)
        loss_GAN_A2B = self.criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))
        pred_fake_A = self.D_A(fake_A)
        loss_GAN_B2A = self.criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle loss
        recovered_A = self.G_B2A(fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)
        recovered_B = self.G_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)

        # Total generator loss
        loss_G = (
            loss_GAN_A2B
            + loss_GAN_B2A
            + self.hparams.lambda_cycle * (loss_cycle_ABA + loss_cycle_BAB)
            + self.hparams.lambda_identity * (loss_identity_A + loss_identity_B)
        )

        self.log("train/loss_G", loss_G, prog_bar=True, on_step=True, on_epoch=True)

        opt_G.zero_grad()
        self.manual_backward(loss_G)
        opt_G.step()

        # Train Discriminators
        # Discriminator A
        pred_real_A = self.D_A(real_A)
        loss_D_real_A = self.adversarial_loss(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = self.D_A(fake_A.detach())
        loss_D_fake_A = self.adversarial_loss(
            pred_fake_A, torch.zeros_like(pred_fake_A)
        )
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

        # Discriminator B
        pred_real_B = self.D_B(real_B)
        loss_D_real_B = self.adversarial_loss(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = self.D_B(fake_B.detach())
        loss_D_fake_B = self.adversarial_loss(
            pred_fake_B, torch.zeros_like(pred_fake_B)
        )
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        # Total discriminator loss
        loss_D = loss_D_A + loss_D_B

        self.log("train/loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=True)

        opt_D.zero_grad()
        self.manual_backward(loss_D)
        opt_D.step()

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
            real_A = next(iter(self.trainer.datamodule.train_dataloader()))["A"][:4]
            real_B = next(iter(self.trainer.datamodule.train_dataloader()))["B"][:4]
            fake_B = self.G_A2B(real_A)
            fake_A = self.G_B2A(real_B)

            grid = make_grid(
                torch.cat((real_A, fake_B, real_B, fake_A), dim=0),
                nrow=4,
                normalize=True,
            )

            self.logger.experiment.add_image(
                "Generated_Images", grid, self.current_epoch
            )

        if self.current_epoch % 10 == 0 and self.current_epoch != 0:
            real_A = torch.cat(self.fid_real_A)
            fake_B = torch.cat(self.fid_fake_B)
            real_B = torch.cat(self.fid_real_B)
            fake_A = torch.cat(self.fid_fake_A)

            self.fid.update(real_A, real=True)
            self.fid.update(fake_B, real=False)
            fid_score_A2B = self.fid.compute().item()

            self.fid.reset()

            self.fid.update(real_B, real=True)
            self.fid.update(fake_A, real=False)
            fid_score_B2A = self.fid.compute().item()

            self.log("train/FID_A2B", fid_score_A2B, prog_bar=True)
            self.log("train/FID_B2A", fid_score_B2A, prog_bar=True)

            self.fid_real_A.clear()
            self.fid_fake_B.clear()
            self.fid_real_B.clear()
            self.fid_fake_A.clear()

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [opt_G, opt_D], []
