#!/bin/bash
#SBATCH --job-name=gan_exploration    
#SBATCH --partition=clara
#SBATCH --output=output_%j.txt    
#SBATCH --error=error_%j.txt 
#SBATCH --nodes=2         
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=4            
#SBATCH --mem=48G                               
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=john.jaenckel@icloud.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --signal=SIGUSR1@90

# Load necessary modules
source python_env/bin/activate

# Run your training script
srun python main.py --model_name=$MODEL_NAME --g_type=$G_TYPE --d_type=$D_TYPE
