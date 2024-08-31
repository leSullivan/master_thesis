import os
import torch
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.create_training_dataset import sample_images
from src.data_pipeline import UnpairedImageDataModule
from models import CGAN, CycleGAN

from src.config import (
    SEED,
    MODEL_TYPE,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    BETA1,
    BETA2,
    IMG_H,
    IMG_W,
    NDF,
    NGF,
    N_DOWNSAMPLING,
    NORM_TYPE,
    DISCRIMINATOR_TYPE,
    GENERATOR_TYPE,
    ND_LAYERS,
    LAMBDA_L1,
)


def main(args):

    data_module = UnpairedImageDataModule(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        img_h=args.img_h,
        img_w=args.img_w,
    )

    if args.model_name.lower() == "cgan":
        GAN = CGAN(**vars(args))
    elif args.model_namelower() == "cyclegan":
        GAN = CycleGAN(**vars(args))
    else:
        raise NotImplementedError(f"Invalid model name: {args.model}")

    if args.checkpoint_version_name is not None:
        path = _get_checkpoint_path(args.model_name, args.checkpoint_path)
        model = GAN.load_from_checkpoint(path).to(torch.float32)
    else:
        model = GAN.to(torch.float32)

    logger = TensorBoardLogger("tb_logs", name=args.model_name)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=50,
        filename="{epoch:04d}",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "mps",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)


def _get_checkpoint_path(model_name, version_name):
    if version_name is None:
        return None

    path = os.path.join(
        "tb_logs",
        model_name,
        version_name,
        "last.ckpt",
    )

    assert os.path.exists(path), f"Invalid checkpoint path: {path}"

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN model")

    # Training Setup
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_TYPE,
    )
    parser.add_argument(
        "--img-w",
        type=int,
        default=IMG_W,
    )
    parser.add_argument(
        "--img-h",
        type=int,
        default=IMG_H,
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default=NORM_TYPE,
    )

    # generator
    parser.add_argument(
        "--generator_type",
        type=str,
        default=GENERATOR_TYPE,
    )

    parser.add_argument(
        "--ngf", type=int, default=NGF, help="Number of generator filters"
    )

    parser.add_argument(
        "--n_downsampling",
        type=int,
        default=N_DOWNSAMPLING,
    )

    # discriminator
    parser.add_argument(
        "--discriminator_type",
        type=str,
        default=DISCRIMINATOR_TYPE,
    )

    parser.add_argument("--ndf", type=int, default=NDF)

    parser.add_argument(
        "--nd_layers",
        type=str,
        default=ND_LAYERS,
    )

    # training
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta1", type=float, default=BETA1)
    parser.add_argument("--beta2", type=float, default=BETA2)
    parser.add_argument(
        "--lamba_identity",
        type=float,
        default=LAMBDA_L1,
    )

    # checkpoint
    parser.add_argument(
        "--checkpoint_version_name",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if False:
        print("Generating sample images...")
        sample_images()
        print("Sample images generated successfully.")
    else:
        pl.seed_everything(SEED)
        torch.use_deterministic_algorithms(True)
        main(args)
