import os
import gc
import torch
import argparse
import pytorch_lightning as pl

from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.create_training_dataset import sample_images
from src.data_pipeline import UnpairedImageDataModule
from src.frameworks import CGAN, CycleGAN, TurboCycleGAN
from src.models import SDTurboGenerator

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
    N_DOWNSAMPLING_RES_NET,
    N_DOWNSAMPLING_U_NET,
    NORM_TYPE,
    D_TYPE,
    D_USE_SIGMOID,
    G_TYPE,
    ND_LAYERS,
    LAMBDA_L1,
    LAMBDA_CYCLE,
    LAMBDA_GAN,
    LAMBDA_PERCEPTUAL,
    PROMPT_BG,
    PROMPT_FENCE,
    CROP,
    WEIGHT_DECAY,
    ADAM_EPS,
)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/work/jj17qosa-gan/")

torch.cuda.empty_cache()


def main(args):

    data_module = UnpairedImageDataModule(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        img_h=args.img_h,
        img_w=args.img_w,
        crop=CROP,
    )

    if args.model_name.lower() == "cgan":
        GAN = CGAN(**vars(args))
    elif args.model_name.lower() == "cyclegan":
        GAN = CycleGAN(**vars(args))
    elif args.model_name.lower() == "turbo_cyclegan":
        GAN = TurboCycleGAN(**vars(args))
    else:
        raise NotImplementedError(f"Invalid model name: {args.model}")

    if args.checkpoint_version_name is not None:
        assert args.model_name in args.checkpoint_path, "Checkpoint model name mismatch"
        path = _get_checkpoint_path(args.model_name, args.checkpoint_path)
        model = GAN.load_from_checkpoint(path)
    else:
        model = GAN

    logger = TensorBoardLogger(
        CHECKPOINT_PATH,
        name=(
            f"{args.model_name}_{args.g_type}_{args.d_type}_ngf{args.ngf}_perc{args.lambda_perceptual}_cyc{args.lambda_cycle}_({args.img_h},{args.img_w})"
            if not CROP
            else f"CROP_{args.model_name}_{args.g_type}_{args.d_type}_ngf{args.ngf}_perc{args.lambda_perceptual}_cyc{args.lambda_cycle}"
        ),
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=100,
        filename="{epoch:04d}",
        save_last=True,
    )

    if args.model_name.lower() == "turbo_cyclegan":
        trainer = pl.Trainer(
            strategy=FSDPStrategy(auto_wrap_policy={SDTurboGenerator}),
            max_epochs=args.num_epochs,
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            precision="16-mixed",
            max_epochs=args.num_epochs,
            logger=logger,
            callbacks=[lr_monitor, checkpoint_callback],
        )

    trainer.fit(model, datamodule=data_module)


def _get_checkpoint_path(model_name, version_name):
    if version_name is None:
        return None

    path = os.path.join(
        CHECKPOINT_PATH,
        model_name,
        version_name,
        "last.ckpt",
    )

    assert os.path.exists(path), f"Invalid checkpoint path: {path}"

    return path


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser(description="Train a GAN model")

    # Training Setup
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_TYPE,
    )
    calc_scores = os.getenv("CALC_SCORES", "True")
    parser.add_argument(
        "--calculate_scores_during_training",
        type=bool,
        default=calc_scores == "True",
    )
    parser.add_argument(
        "--img_w",
        type=int,
        default=IMG_W,
    )
    parser.add_argument(
        "--img_h",
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
        "--g_type",
        type=str,
        default=G_TYPE,
    )

    parser.add_argument(
        "--ngf", type=int, default=NGF, help="Number of generator filters"
    )

    parser.add_argument(
        "--n_downsampling",
        type=int,
        default=(
            N_DOWNSAMPLING_RES_NET if "resnet" in G_TYPE else N_DOWNSAMPLING_U_NET
        ),
    )

    parser.add_argument(
        "--prompt_fence",
        type=str,
        default=PROMPT_FENCE,
    )
    parser.add_argument(
        "--prompt_bg",
        type=str,
        default=PROMPT_BG,
    )

    # discriminator
    parser.add_argument(
        "--d_type",
        type=str,
        default=D_TYPE,
    )

    parser.add_argument("--ndf", type=int, default=NDF)

    parser.add_argument(
        "--nd_layers",
        type=str,
        default=ND_LAYERS,
    )
    parser.add_argument(
        "--d_use_sigmoid",
        type=bool,
        default=D_USE_SIGMOID,
    )

    # training
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--adam_eps", type=float, default=ADAM_EPS)
    parser.add_argument("--beta1", type=float, default=BETA1)
    parser.add_argument("--beta2", type=float, default=BETA2)
    parser.add_argument(
        "--lambda_identity",
        type=float,
        default=LAMBDA_L1,
    )
    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=LAMBDA_CYCLE,
    )
    parser.add_argument(
        "--lambda_gan",
        type=float,
        default=LAMBDA_GAN,
    )
    parser.add_argument(
        "--lambda_perceptual",
        type=float,
        default=LAMBDA_PERCEPTUAL,
    )

    parser.add_argument(
        "--crop",
        type=bool,
        default=CROP,
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
        main(args)
