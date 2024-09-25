#!/bin/bash
for model_name in cgan; do
   for g_type in resnet-9 unet-7; do 
      for d_type in basic patch pixel; do 
         sbatch --export=MODEL_NAME=$model_name,G_TYPE=$g_type,D_TYPE=$d_type slurm_template.sh
      done
   done
done

sbatch --export=MODEL_NAME=cgan,G_TYPE=resnet-6,D_TYPE=pixel slurm_template.sh