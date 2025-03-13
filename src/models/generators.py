import torch
import copy
import pytorch_lightning as pl
import torch.nn as nn
import functools

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig

from src.config import IMG_CH
from src.utils import get_norm_layer, get_device



class Generator(pl.LightningModule):
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

        elif g_type == "resnet-skip-con":
            self.model = ResNetGenerator(
                kwargs["ngf"],
                kwargs["n_downsampling"],
                norm_layer,
                input_nc,
                use_skip_connections=True,
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

        elif g_type == "unet_128":
            self.model = UNetGenerator(
                input_nc,
                input_nc,
                7,
                kwargs["ngf"],
                norm_layer,
                use_dropout=False,
            )

        elif g_type == "unet_256":
            self.model = UNetGenerator(
                input_nc,
                input_nc,
                8,
                kwargs["ngf"],
                norm_layer,
                use_dropout=False,
            )

        elif g_type == "unet_512":
            self.model = UNetGenerator(
                input_nc,
                input_nc,
                9,
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


# Diffusion Generator
# https://github.com/GaParmar/img2img-turbo/blob/main/src/cyclegan_turbo.py
class SDTurboGenerator(pl.LightningModule):
    def __init__(
        self,
        prompt_bg,
        prompt_fence,
        model="sd-turbo",
    ):
        super(SDTurboGenerator, self).__init__()
        # Let Lightning handle the device management instead of manual assignment
        # Remove the get_device() call as Lightning will handle this
        device = get_device()

        self.unet = initialize_unet(model=model)
        self.vae = initialize_vae(model=model)
        self.scheduler = get_1step_sched(device, model=model)  # Remove device parameter

        self.encoder = VAE_encode(self.vae, copy.deepcopy(self.vae))
        self.decoder = VAE_decode(self.vae, copy.deepcopy(self.vae))

        tokenizer = AutoTokenizer.from_pretrained(
            f"stabilityai/{model}",
            subfolder="tokenizer",
            use_fast=False,
        )

        text_encoder = CLIPTextModel.from_pretrained(
            f"stabilityai/{model}", subfolder="text_encoder"
        )

        # Move token processing to device-aware context
        fence_prompt_tokens = tokenizer(
            prompt_fence,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        bg_prompt_tokens = tokenizer(
            prompt_bg,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # Register embeddings as buffers so Lightning can handle their device placement
        self.register_buffer(
            "bg2fence_emb", text_encoder(fence_prompt_tokens.unsqueeze(0))[0].detach()
        )
        self.register_buffer(
            "fence2bg_emb", text_encoder(bg_prompt_tokens.unsqueeze(0))[0].detach()
        )

    def forward(self, x, direction):
        batch_size = x.shape[0]
        timesteps = torch.tensor(
            [self.scheduler.config.num_train_timesteps - 1] * batch_size,
            device=x.device,  # Use input tensor's device
        ).long()

        caption_emb_base = (
            self.bg2fence_emb if direction == "Bg2Fence" else self.fence2bg_emb
        )
        # No need for explicit device movement as buffers follow module's device
        caption_emb = caption_emb_base.repeat(batch_size, 1, 1)

        x_enc, current_down_blocks = self.encoder(x, direction=direction)
        model_pred = self.unet(
            x_enc,
            timesteps,
            encoder_hidden_states=caption_emb,
        ).sample

        x_out = torch.stack(
            [
                self.scheduler.step(
                    model_pred[i], timesteps[i], x_enc[i], return_dict=True
                ).prev_sample
                for i in range(batch_size)
            ]
        )

        x_out_decoded = self.decoder(
            x_out, direction=direction, current_down_blocks=current_down_blocks
        )
        return x_out_decoded


# https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class ResNetGenerator(pl.LightningModule):
    def __init__(
        self,
        ngf,
        n_downsampling,
        norm_layer,
        input_channels,
        n_blocks=6,
        padding_type="reflect",
        use_skip_connections=False,
    ):
        assert n_blocks >= 0
        super(ResNetGenerator, self).__init__()
        self.use_skip_connections = use_skip_connections
        activation = nn.ReLU(True)

        # Initial layers
        self.initial_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        )

        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2**i
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1
                    ),
                    norm_layer(ngf * mult * 2),
                    activation,
                )
            )

        # ResNet blocks
        self.resnet_blocks = nn.ModuleList()
        mult = 2**n_downsampling
        for i in range(n_blocks):
            self.resnet_blocks.append(
                ResnetBlock(
                    ngf * mult, padding_type=padding_type, norm_layer=norm_layer
                )
            )

        # Upsample layers
        self.upsample_layers = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ngf * mult if not self.use_skip_connections else ngf * mult * 2,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    activation,
                )
            )

        # Final layers
        self.final_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, input_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.initial_layers(input)

        # Downsample with skip connection storage
        downsample_outputs = []
        for layer in self.downsample_layers:
            x = layer(x)
            if self.use_skip_connections:
                downsample_outputs.append(x)  # Store outputs for skip connections

        # ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)

        # Upsample with optional skip connections
        for i, layer in enumerate(self.upsample_layers):
            if self.use_skip_connections:
                skip = downsample_outputs[-(i + 1)]  # Get corresponding skip connection
                x = torch.cat(
                    [x, skip], dim=1
                )  # Concatenate along the channel dimension
            x = layer(x)

        return self.final_layers(x)


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNetGenerator(pl.LightningModule):
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
class ResnetBlock(pl.LightningModule):
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
class UnetSkipConnectionBlock(pl.LightningModule):

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


class VAE_encode(pl.LightningModule):
    def __init__(self, vae_BgToFence, vae_Fence2Bg):
        super(VAE_encode, self).__init__()
        self.vae_BgToFence = vae_BgToFence
        self.vae_Fence2Bg = vae_Fence2Bg

    def forward(self, x, direction):
        assert direction in ["Bg2Fence", "Fence2Bg"]

        _vae = self.vae_BgToFence if direction == "Bg2Fence" else self.vae_Fence2Bg

        latent = _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor

        return latent, _vae.encoder.current_down_blocks


class VAE_decode(pl.LightningModule):
    def __init__(self, vae_BgToFence, vae_Fence2Bg):
        super(VAE_decode, self).__init__()
        self.vae_BgToFence = vae_BgToFence
        self.vae_Fence2Bg = vae_Fence2Bg

    def forward(self, x, direction, current_down_blocks):
        assert direction in ["Bg2Fence", "Fence2Bg"]
        _vae = self.vae_BgToFence if direction == "Bg2Fence" else self.vae_Fence2Bg

        if current_down_blocks is not None:
            _vae.decoder.incoming_skip_acts = current_down_blocks
        else:
            print("Warning: current_down_blocks is None")

        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank=8, return_lora_module_names=False, model="sd-turbo"):
    # load unet
    unet = UNet2DConditionModel.from_pretrained(
        f"stabilityai/{model}", subfolder="unet"
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
        return [
            *l_target_modules_encoder,
            *l_target_modules_decoder,
            *l_modules_others,
        ]
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False, model="sd-turbo"):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/{model}", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(
        512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
    ).requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(
        256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
    ).requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(
        128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
    ).requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(
        128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
    ).requires_grad_(True)
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
        return [*l_vae_target_modules]
    else:
        return vae


def get_1step_sched(device, model="sd-turbo"):

    noise_scheduler_1step = DDPMScheduler.from_pretrained(
        f"stabilityai/{model}", subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device=device)
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
    # Initial convolution
    sample = self.conv_in(sample)

    # Store the data type for upscaling
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    # Middle block (with optional latent embeddings)
    sample = self.mid_block(sample, latent_embeds)

    # Convert sample to the upscale data type
    sample = sample.to(upscale_dtype)

    if not self.ignore_skip:
        # Define skip convolutions
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]

        # Upsampling with skip connections
        for idx, up_block in enumerate(self.up_blocks):
            # Get the skip connection from previous layers
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)

            # Ensure the skip connection matches the sample size before adding
            if skip_in.size(2) != sample.size(2) or skip_in.size(3) != sample.size(3):
                # print(f"Resizing sample from {sample.size()} to {skip_in.size()}")
                sample = nn.functional.interpolate(
                    sample,
                    size=skip_in.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Add the skip connection to the sample
            sample = sample + skip_in

            # Apply upsampling block
            sample = up_block(sample, latent_embeds)
    else:
        # If no skip connections, just pass through upsampling blocks
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    # Post-processing
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)

    # Final activation and output convolution
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return sample
