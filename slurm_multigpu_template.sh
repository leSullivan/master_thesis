#!/bin/bash
#SBATCH --job-name=gan_exploration    
#SBATCH --partition=clara
#SBATCH --output=slurm_res/output_%j.txt    
#SBATCH --error=slurm_res/error_%j.txt 
#SBATCH --nodes=1         
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus=4            
#SBATCH --mem=200GB                            
#SBATCH --gres=gpu:v100:4
#SBATCH --time=48:00:00

# Load necessary modules
module load NCCL

#Overwrite SLURM Env
export SLURM_TRES_PER_TASK="cpu=4"

# Set up NCCL environment
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1

# Set up distributed training environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Activate virtual environment
source python_env/bin/activate

# Run your training script
srun python main.py \
    --model_name=$MODEL_NAME \
    --g_type=$G_TYPE \
    --d_type=$D_TYPE \
    --lambda_cycle=$LAMBDA_CYCLE \
    --lambda_perceptual=$LAMBDA_PERCEPTUAL \
    --lambda_gan=$LAMBDA_GAN \
    --ngf=$NGF \
    --crop=$CROP