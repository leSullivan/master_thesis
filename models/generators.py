import torch
import copy
import torch.nn as nn
import functools

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

from src.config import IMG_CH

from .utils import get_norm_layer


class Generator(nn.Module):
    def __init__(
        self,
        g_type,
        input_nc=IMG_CH,
        **kwargs,
    ):
        super(Generator, self).__init__()

        norm_layer = get_norm_layer(kwargs["norm_type"])

        if g_type == "resnet-6":
            self.model = ResNetGenerator(
                kwargs["ngf"], kwargs["n_downsampling"], norm_layer, input_nc
            )

        elif g_type == "resnet-9":
            self.model = ResNetGenerator(
                kwargs["ngf"],
                kwargs["n_downsampling"],
                norm_layer,
                input_nc,
                n_blocks=9,
            )

        elif g_type == "unet":
            self.model = UNetGenerator(
                input_nc,
                input_nc,
                kwargs["n_downsampling"],
                kwargs["ngf"],
                norm_layer,
                use_dropout=False,
            )

        elif g_type == "unet-dropout":
            self.model = UNetGenerator(
                input_nc,
                input_nc,
                kwargs["n_downsampling"],
                kwargs["ngf"],
                norm_layer,
                use_dropout=True,
            )
        else:
            raise ValueError(f"Generator type '{g_type}' is not recognized.")

    def forward(self, x, **kwargs):
        output = self.model.forward(x, **kwargs)
        return output


# https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class ResNetGenerator(nn.Module):
    def __init__(
        self,
        ngf,
        n_downsampling,
        norm_layer,
        input_channels,
        n_blocks=6,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(ResNetGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult, padding_type=padding_type, norm_layer=norm_layer
                )
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, input_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        n_downsampling,
        ngf,
        norm_layer,
        use_dropout=False,
    ):
        super(UNetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for _ in range(
            n_downsampling - 5
        ):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input):
        return self.model(input)


# https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UnetSkipConnectionBlock(nn.Module):

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


# Diffusion Generator
# https://github.com/GaParmar/img2img-turbo/blob/main/src/cyclegan_turbo.py
class SDTurboGenerator(torch.nn.Module):
    def __init__(
        self,
        device,
        prompt_bg,
        prompt_fence,
    ):
        super(SDTurboGenerator, self).__init__()
        self.unet = initialize_unet().to(device)
        # unet.enable_xformers_memory_efficient_attention()
        # unet.enable_gradient_checkpointing()
        vae = initialize_vae(device).to(device)
        self.vae = vae

        self.scheduler = get_1step_sched(device=device)

        self.encoder = VAE_encode(vae, copy.deepcopy(vae)).to(device)
        self.decoder = VAE_decode(vae, copy.deepcopy(vae)).to(device)

        self.timesteps = torch.tensor([999], device=device).long()

        tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="tokenizer",
            use_fast=False,
        ).to(device)

        text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).to(device)

        fence_prompt_tokens = tokenizer(
            prompt_fence,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        self.bg2fence_emb = text_encoder(fence_prompt_tokens.to(device).unsqueeze(0))[
            0
        ].detach()

        bg_prompt_tokens = tokenizer(
            prompt_bg,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        self.fence2bg_emb = text_encoder(bg_prompt_tokens.to(device).unsqueeze(0))[
            0
        ].detach()

    # def get_traininable_params(self):
    #     # add all unet parameters
    #     params_gen = list(self.unet.conv_in.parameters())
    #     self.unet.conv_in.requires_grad_(True)
    #     self.unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    #     for n, p in self.unet.named_parameters():
    #         if "lora" in n and "default" in n:
    #             assert p.requires_grad
    #             params_gen.append(p)

    #     # add all vae_a2b parameters
    #     for n, p in self.vae.named_parameters():
    #         if "lora" in n and "vae_skip" in n:
    #             assert p.requires_grad
    #             params_gen.append(p)
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_1.parameters())
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_2.parameters())
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_3.parameters())
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_4.parameters())

    #     # add all vae_b2a parameters
    #     for n, p in self.vae.named_parameters():
    #         if "lora" in n and "vae_skip" in n:
    #             assert p.requires_grad
    #             params_gen.append(p)
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_1.parameters())
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_2.parameters())
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_3.parameters())
    #     params_gen = params_gen + list(self.vae.decoder.skip_conv_4.parameters())
    #     return params_gen

    def forward(self, x, direction):
        batch_size = x.shape[0]

        caption_emb_base = (
            self.bg2fence_emb if direction == "Bg2Fence" else self.fence2bg_emb
        )
        # check dtype ! (to  tochfloat 32)
        caption_emb = caption_emb_base.repeat(batch_size, 1, 1)

        # encode to latent space
        x_enc = self.encoder(x, direction=direction).to(x.dtype)
        # duffision steps
        model_pred = self.unet(
            x_enc,
            self.timesteps,
            encoder_hidden_states=caption_emb,
        ).sample
        x_out = torch.stack(
            [
                self.scheduler.step(
                    model_pred[i], self.timesteps[i], x_enc[i], return_dict=True
                ).prev_sample
                for i in range(batch_size)
            ]
        )
        # decode to image space
        x_out_decoded = self.decoder(x_out, direction=direction)
        return x_out_decoded


class VAE_encode(nn.Module):
    def __init__(self, vae_BgToFence, vae_Fence2Bg):
        super(VAE_encode, self).__init__()
        self.vae_BgToFence = vae_BgToFence
        self.vae_Fence2Bg = vae_Fence2Bg

    def forward(self, x, direction):
        assert direction in ["Bg2Fence", "Fence2Bg"]
        if direction == "Bg2Fence":
            _vae = self.vae_BgToFence
        else:
            _vae = self.vae_Fence2Bg
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae_BgToFence, vae_Fence2Bg):
        super(VAE_decode, self).__init__()
        self.vae_BgToFence = vae_BgToFence
        self.vae_Fence2Bg = vae_Fence2Bg

    def forward(self, x, direction):
        assert direction in ["Bg2Fence", "Fence2Bg"]
        if direction == "Bg2Fence":
            _vae = self.vae_BgToFence
        else:
            _vae = self.vae_Fence2Bg
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank=8, return_lora_module_names=False):
    # load unet
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sd-turbo", subfolder="unet"
    )
    # frezze model params
    unet.requires_grad_(False)
    unet.train()
    # get lora relevant module names
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv_out",
        "proj_out",
        "proj_in",
        "ff.net.2",
        "ff.net.0.proj",
    ]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    # init lora confogs for encoder, decoder and others
    lora_conf_encoder = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_encoder,
        lora_alpha=rank,
    )
    lora_conf_decoder = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_target_modules_decoder,
        lora_alpha=rank,
    )
    lora_conf_others = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_modules_others,
        lora_alpha=rank,
    )
    # add lora adapters
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])

    if return_lora_module_names:
        return (
            unet,
            l_target_modules_encoder,
            l_target_modules_decoder,
            l_modules_others,
        )
    else:
        return unet


def initialize_vae(device, rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = (
        torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .to(device)
        .requires_grad_(True)
    )
    vae.decoder.skip_conv_2 = (
        torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .to(device)
        .requires_grad_(True)
    )
    vae.decoder.skip_conv_3 = (
        torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .to(device)
        .requires_grad_(True)
    )
    vae.decoder.skip_conv_4 = (
        torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        .to(device)
        .requires_grad_(True)
    )
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "skip_conv_1",
        "skip_conv_2",
        "skip_conv_3",
        "skip_conv_4",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    vae_lora_config = LoraConfig(
        r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules
    )
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


def get_1step_sched(device):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(
        "stabilityai/sd-turbo", subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device=device)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(
        device
    )
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample
