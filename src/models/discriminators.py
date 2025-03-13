import functools
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from src.config import IMG_CH
from src.utils import get_norm_layer, get_device


class Discriminator(pl.LightningModule):
    def __init__(
        self,
        ndf,
        nd_layers,
        norm_type,
        d_type,
        d_use_sigmoid,
        input_channels=IMG_CH,
    ):
        super(Discriminator, self).__init__()
        self.d_use_sigmoid = d_use_sigmoid
        self.d_type = d_type

        norm_layer = get_norm_layer(norm_type)

        if d_type == "basic":
            self.model = NLayerDiscriminator(
                ndf,
                nd_layers,
                norm_layer,
                input_channels,
                use_patches=False,
            )

        elif d_type == "patch":
            self.model = NLayerDiscriminator(
                ndf,
                nd_layers,
                norm_layer,
                input_channels,
                use_sigmoid=d_use_sigmoid,
            )

        elif d_type == "pixel":
            self.model = PixelDiscriminator(input_channels, ndf, norm_layer)

        elif d_type == "vagan":
            import vision_aided_loss

            device = get_device()

            self.model = vision_aided_loss.Discriminator(
                cv_type="clip", loss_type="multilevel_sigmoid_s", device=device
            )
            self.model.cv_ensemble.requires_grad_(False)
        else:
            raise ValueError(f"Discriminator type '{d_type}' is not recognized.")

    def forward(self, x, **kwargs):
        if self.d_type == "vagan":
            output = self.model(x, **kwargs)
        else:
            output = self.model(x)
        return output


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class NLayerDiscriminator(pl.LightningModule):
    def __init__(
        self, ndf, n_layers, norm_layer, input_nc, use_sigmoid=False, use_patches=True
    ):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True),
        ]

        if use_patches:
            sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]
            if use_sigmoid:
                sequence += [nn.Sigmoid()]
        else:
            sequence += [nn.AdaptiveAvgPool2d(1)]
            sequence += [nn.Flatten()]
            sequence += [nn.Linear(nf, 1)]
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class PixelDiscriminator(pl.LightningModule):
    def __init__(self, input_nc, ndf, norm_layer):

        super(PixelDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
