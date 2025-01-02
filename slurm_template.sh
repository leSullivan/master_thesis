#!/bin/bash
#SBATCH --job-name=gan_exploration    
#SBATCH --partition=clara
#SBATCH --output=slurm_res/output_%j.txt    
#SBATCH --error=slurm_res/error_%j.txt 
#SBATCH --nodes=1         
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=4            
#SBATCH --mem=64G                               
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@90

module load CUDA/12.4.0

source python_env/bin/activate
unset SLURM_TRES_PER_TASK

srun python main.py \
    --model_name=$MODEL_NAME \
    --g_type=$G_TYPE \
    --d_type=$D_TYPE \
    --lambda_cycle=$LAMBDA_CYCLE \
    --lambda_perceptual=$LAMBDA_PERCEPTUAL \
    --lambda_gan=$LAMBDA_GAN \
    --ngf=$NGF \
    --crop=$CROP \
    --batch_size=1 \
    --img_h=$IMG_H \
    --img_w=$IMG_W
