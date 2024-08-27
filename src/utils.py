import os
import torch
from torch.nn import Module
from datetime import datetime


def save_model_checkpoint(name: str, arch: str, strategy: str, model: Module):
    model_checkpoint_path = os.path.join(
        "model_checkpoints",
        arch,
        strategy,
        datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
    )
    os.makedirs(model_checkpoint_path, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(
            model_checkpoint_path,
            f"{name}_model.pth",
        ),
    )


def weights_init(m: Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
