import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

from src.create_training_dataset import sample_images
from src.data_pipeline import UnpairedImageDataModule
from src.gan_training import UnpairedImageModel
from src.models import Generator, Discriminator

from src.config import SEED, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, BETA1, BETA2


def main():
    data_module = UnpairedImageDataModule(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    generator = Generator()
    discriminator = Discriminator()

    model = UnpairedImageModel(
        generator=generator,
        discriminator=discriminator,
        lr=LR,
        beta1=BETA1,
        beta2=BETA2,
    ).to(torch.float32)

    logger = TensorBoardLogger("tb_logs", name="unpaired_image_translation")

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
