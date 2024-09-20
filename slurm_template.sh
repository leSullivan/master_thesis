#!/bin/bash
#SBATCH --job-name=gan_exploration    
#SBATCH --partition=clara
#SBATCH --output=slurm_res/output_%j.txt    
#SBATCH --error=slurm_res/error_%j.txt          
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=4            
#SBATCH --mem=32G                               
#SBATCH --gres=gpu:v100:1         
#SBATCH --time=48:00:00
#SBATCH --mail-user=john.jaenckel@icloud.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
source .venv/bin/activate

# Run your training script
srun python main.py --model_name=$MODEL_NAME