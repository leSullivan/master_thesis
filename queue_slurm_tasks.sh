#!/bin/bash
for model_name in cgan; do
   for g_type in resnet-6 resnet-9 unet-512  ; do 
      for lambda_cycle in 0.1 1 10 20; do 
         sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,D_TYPE=$d_type slurm_template.sh
      done
   done
done