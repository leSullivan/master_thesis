#!/bin/bash
#SBATCH --job-name=gan_exploration    
#SBATCH --output=output_%j.txt    
#SBATCH --error=error_%j.txt          
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=4            
#SBATCH --mem=32G                     
#SBATCH --partition=gpu            
#SBATCH --gpus:v100:2              
#SBATCH --time=48:00:00
#SBATCH --mail-user=john.jaenckel@icloud.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules (example)
module r default

# Run your training script
srun python train.py --d_type=$D_TYPE --g_type=$G_TYPE
