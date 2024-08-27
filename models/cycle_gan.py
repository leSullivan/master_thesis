import torch
from torch import nn

from src.config import IMG_CH

import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=True,
        use_dropout=False,
        norm_type="instance",
    ):
        super(UNetBlock, self).__init__()

        if norm_type.lower() == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_type.lower() == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        conv_layer = (
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride=2,
                padding=1,
                bias=False,
            )
            if downsample
            else nn.ConvTranspose2d(
                in_channels, out_channels, 4, stride=2, padding=1, bias=False
            )
        )

        act_layer = (
            nn.LeakyReLU(0.2, inplace=True) if downsample else nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(conv_layer, norm_layer(out_channels), act_layer)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class UNetGenerator(nn.Module):
    def __init__(
        self, input_channels=3, output_channels=3, ngf=64, norm_type="instance"
    ):
        super(UNetGenerator, self).__init__()

        self.down1 = UNetBlock(
            input_channels, ngf, downsample=True, norm_type=norm_type
        )
        self.down2 = UNetBlock(ngf, ngf * 2, downsample=True, norm_type=norm_type)
        self.down3 = UNetBlock(ngf * 2, ngf * 4, downsample=True, norm_type=norm_type)
        self.down4 = UNetBlock(ngf * 4, ngf * 8, downsample=True, norm_type=norm_type)
        self.down5 = UNetBlock(ngf * 8, ngf * 8, downsample=True, norm_type=norm_type)
        self.down6 = UNetBlock(ngf * 8, ngf * 8, downsample=True, norm_type=norm_type)
        self.down7 = UNetBlock(ngf * 8, ngf * 8, downsample=True, norm_type=norm_type)
        self.down8 = UNetBlock(ngf * 8, ngf * 8, downsample=True, norm_type=norm_type)

        self.up1 = UNetBlock(
            ngf * 8, ngf * 8, downsample=False, norm_type=norm_type, use_dropout=True
        )
        self.up2 = UNetBlock(
            ngf * 16, ngf * 8, downsample=False, norm_type=norm_type, use_dropout=True
        )
        self.up3 = UNetBlock(
            ngf * 16, ngf * 8, downsample=False, norm_type=norm_type, use_dropout=True
        )
        self.up4 = UNetBlock(ngf * 16, ngf * 8, downsample=False, norm_type=norm_type)
        self.up5 = UNetBlock(ngf * 16, ngf * 4, downsample=False, norm_type=norm_type)
        self.up6 = UNetBlock(ngf * 8, ngf * 2, downsample=False, norm_type=norm_type)
        self.up7 = UNetBlock(ngf * 4, ngf, downsample=False, norm_type=norm_type)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=IMG_CH):
        super(PatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)
