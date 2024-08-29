import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

from src.create_training_dataset import sample_images
from src.data_pipeline import UnpairedImageDataModule
from models.baseline import Generator, Discriminator, cGAN

from src.config import SEED, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, BETA1, BETA2

_architecture_mapping = {
    "baseline": cGAN,
}

_network_mapppin = {
    "baseline": [Generator, Discriminator],
}


def main(model_name="baseline"):
    data_module = UnpairedImageDataModule(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    _GAN = _architecture_mapping[model_name]
    _Generator, _Discriminator = _network_mapppin[model_name]

    generator = _Generator()
    discriminator = _Discriminator()

    model = _GAN(
        generator=generator,
        discriminator=discriminator,
    ).to(torch.float32)

    logger = TensorBoardLogger("tb_logs", name=model_name)

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        log_every_n_steps=1,
        logger=logger,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.use_deterministic_algorithms(True)
    main()
