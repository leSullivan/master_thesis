import torch
import functools
import torchvision
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn


# needed as device isnt available in module.init()
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_trainable_modules(lightning_module):
    """
    Returns a list of trainable modules from a PyTorch Lightning module.
    """
    trainable_modules = []
    for _, module in lightning_module.named_modules():
        if any(param.requires_grad for param in module.parameters(recurse=False)):
            trainable_modules.append(module)
    return trainable_modules


def init_weights(net, net_type, init_type="kaiming", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize {net_type} with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_norm_layer(norm_type):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def preprocess_for_fid(x):
    normalized = (x + 1) / 2
    return (normalized * 255).byte()


def init_gan_loss(d_type):
    if d_type == "basic":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.MSELoss()


# https://github.com/GaParmar/img2img-turbo
################################DINO Struct Loss###############################
def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = "block"
    ATTN_KEY = "attn"
    PATCH_IMD_KEY = "patch_imd"
    QKV_KEY = "qkv"
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        # pdb.set_trace()
        self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.device = device
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(
                    block.register_forward_hook(self._get_block_hook())
                )
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(
                    block.attn.attn_drop.register_forward_hook(self._get_attn_hook())
                )
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(
                    block.attn.qkv.register_forward_hook(self._get_qkv_hook())
                )
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(
                    block.attn.register_forward_hook(self._get_patch_imd_hook())
                )

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        input_img = input_img.to(self.device)
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (
            self.get_height_patch_num(input_img_shape)
            * self.get_width_patch_num(input_img_shape)
        )
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(
            1, 2, 0, 3
        )[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(
            1, 2, 0, 3
        )[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(
            1, 2, 0, 3
        )[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map


class DinoStructureLoss:
    def __init__(self):

        device = get_device()

        self.loss = 0.0
        self.n = 0

        self.extractor = VitExtractor(model_name="dino_vitb8", device=device)

    def preprocess(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = (x + 1) / 2

        normalize = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        return normalize(x)

    def update_dino_struct_loss(self, inputs, outputs):

        assert (
            inputs.shape == outputs.shape
        ), "DINO Struct: Input and output lengths do not match"

        inputs = self.preprocess(inputs)
        outputs = self.preprocess(outputs)

        batch_size = inputs.size(0)

        for i in range(batch_size):
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(
                    inputs[i].unsqueeze(0), layer_num=11
                )
            keys_ssim = self.extractor.get_keys_self_sim_from_input(
                outputs[i].unsqueeze(0), layer_num=11
            )
            self.loss += F.mse_loss(keys_ssim, target_keys_self_sim)
            self.n += 1

    def compute(self):
        return self.loss / self.n * 100

    def reset(self):
        self.loss = 0.0
        self.n = 0
