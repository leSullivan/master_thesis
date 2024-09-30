#!/bin/bash
for model_name in cyclegan; do
   for g_type in resnet-6 resnet-9 unet_256; do 
      for lambda_cycle in 0.1 1 10 20; do 
         sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=vagan, slurm_template.sh
      done
   done
done


for model_name in cgan; do
   for g_type in resnet-6 resnet-9; do 
      for d_type in basic patch pixel vagan; do 
         sbatch --export=MODEL_NAME=$model_name,D_TYPE=$d_type,G_TYPE=$g_type,LAMBDA_CYCLE=1 slurm_template.sh
      done
   done
done