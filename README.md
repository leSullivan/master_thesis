# Master Thesis: Synthetic Fence Imagery Using GANs

This repository contains the implementation of various Generative Adversarial Network (GAN) architectures for unpaired image-to-image translation, specifically focused on creating fences on provided landscape images.

### Key Features

- Implementation of multiple GAN architectures:
  - Conditional GAN (CGAN)
  - CycleGAN
  - TurboCycleGAN (with Stable Diffusion Turbo integration)
- PyTorch Lightning-based training framework
- Support for different generator architectures (UNet, ResNet)
- Support for different discriminator architectures
- Data pipeline for unpaired image datasets
- SLURM integration for HPC cluster training
- Comprehensive logging with TensorBoard

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning (v2.5.0)
- Torchvision
- TensorBoard
- LPIPS (Learned Perceptual Image Patch Similarity)
- Transformers
- Diffusers (v0.25.1)
- PEFT (Parameter-Efficient Fine-Tuning)
- Vision-aided loss

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/leSullivan/unpaired_image_synthesis_with_gans.git
   cd unpaired_image_synthesis_with_gans
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file with necessary paths
   - Set `CHECKPOINT_PATH` for model checkpoints

## Data Preparation

The project expects data to be organized in the following directory structure:
```
imgs/
├── backgrounds/       # Background landscape images
├── fences/            # Fence images
├── results/           # Generated results
└── training_data/     # Training dataset
    ├── trainBg/       # Training background images
    ├── trainFence/    # Training fence images
    ├── valBg/         # Validation background images
    └── valFence/      # Validation fence images
```

## Usage

### Training a Model

To train a model, use the `main.py` script with appropriate arguments:

```bash
python main.py --model_name cyclegan --g_type unet_128 --d_type vagan --ngf 64 --img_h 512 --img_w 768 --num_epochs 400
```

Key parameters:
- `--model_name`: Model architecture to use (cgan, cyclegan, turbo_cyclegan)
- `--g_type`: Generator architecture (unet_128, resnet, etc.)
- `--d_type`: Discriminator architecture
- `--ngf`: Number of generator filters
- `--img_h`/`--img_w`: Image dimensions
- `--lambda_perceptual`/`--lambda_cycle`: Loss function weights

### SLURM Integration

For training on HPC clusters, use the provided SLURM scripts:

```bash
./scripts/queue_turbo_cycle_gan.sh
```

Or use the template scripts as a starting point:
```bash
./scripts/slurm_template.sh
./scripts/slurm_multigpu_template.sh
```

### Monitoring Training

To monitor training progress with TensorBoard:

```bash
tensorboard --logdir=/path/to/logs
```

Alternatively, use the provided script to fetch logs from a remote server:
```bash
./scripts/get_tb_logs.sh
```

## Project Structure

- `main.py`: Main entry point for training
- `src/`:
  - `config.py`: Configuration parameters
  - `data_pipeline.py`: Data loading and processing
  - `frameworks/`: GAN implementation frameworks
  - `models/`: Generator and discriminator model implementations
  - `create_training_dataset.py`: Dataset creation utilities
- `scripts/`: Helper scripts for running on SLURM clusters
- `imgs/`: Image directories
- `requirements.txt`: Python dependencies

## Configuration

The default configuration parameters are defined in `src/config.py`. These can be overridden via command-line arguments when running `main.py`.

Key configuration parameters:
- Image dimensions and channels
- Model hyperparameters
- Training hyperparameters (learning rate, batch size, etc.)
- Loss function weights
- Data paths

## Acknowledgements

This project builds upon several open-source implementations and research papers:

### Model Architectures
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) - Implementation of CycleGAN and UNet generator architecture
- [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) - Implementation of ResNet generator architecture
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo) - Implementation of the TurboCycleGAN architecture and DINO Structural Loss

### Papers
- Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. *IEEE International Conference on Computer Vision (ICCV)*.
- Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Wang, T. C., Liu, M. Y., Zhu, J. Y., Tao, A., Kautz, J., & Catanzaro, B. (2018). High-resolution image synthesis and semantic manipulation with conditional gans. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Parmar, G., Zhang, R., & Zhu, J. Y. (2023). Image-to-image Turbo: Iterative Diffusion for High-Quality Image-to-Image Translation. *arXiv preprint arXiv:2312.04451*.

### Libraries
- [PyTorch Lightning](https://www.pytorchlightning.ai/) - Framework for high-performance AI research
- [Diffusers](https://github.com/huggingface/diffusers) - State-of-the-art diffusion models for image generation
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning techniques
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) - Learned Perceptual Image Patch Similarity metric
