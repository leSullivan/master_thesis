import os
import torch
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from src.create_training_dataset import sample_images
from src.data_pipeline import UnpairedImageDataModule
from models.baseline import Generator, Discriminator, cGAN

from src.config import (
    SEED,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    BETA1,
    BETA2,
    IMG_H,
    IMG_W,
)

_architecture_mapping = {
    "baseline": cGAN,
}

_network_mapppin = {
    "baseline": [Generator, Discriminator],
}


def main(args):
    assert (
        args.model_name in _architecture_mapping
    ), f"Invalid model name: {args.model_name}"

    data_module = UnpairedImageDataModule(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        img_h=args.img_h,
        img_w=args.img_w,
    )

    if args.checkpoint_path is not None:
        path = _get_checkpoint_path(args.model_name, args.checkpoint_path)
        _GAN = _architecture_mapping[args.model_name].load_from_checkpoint(path)
    else:
        _GAN = _architecture_mapping[args.model_name]

    _Generator, _Discriminator = _network_mapppin[args.model_name]

    generator = _Generator()
    discriminator = _Discriminator()

    model = _GAN(
        generator=generator,
        discriminator=discriminator,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
    ).to(torch.float32)

    logger = TensorBoardLogger("tb_logs", name=args.model_name)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=50,
        filename="{epoch:04d}",
        save_last=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "mps"

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        logger=logger,
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)


def _get_checkpoint_path(model_name, checkpoint_name):
    if checkpoint_name is None:
        return None

    path = os.path.join(
        "checkpoints",
        model_name,
        checkpoint_name,
    )

    assert os.path.exists(path), f"Invalid checkpoint path: {path}"

    return path


if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.use_deterministic_algorithms(True)

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a GAN model")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta1", type=float, default=BETA1)
    parser.add_argument("--beta2", type=float, default=BETA2)
    parser.add_argument(
        "--img-h",
        type=int,
        default=IMG_H,
    )
    parser.add_argument(
        "--img-w",
        type=int,
        default=IMG_W,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="baseline",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args)
