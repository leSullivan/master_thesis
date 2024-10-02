#!/bin/bash
model_name="cgan"
lambda_cycle="0"

for d_type in patch vagan; do
   for g_type in resnet-6 unet_256; do 
      for lambda_perceptual in 1 5 10; do 
         for lambda_gan in 0.5 1 2; do 
            sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan slurm_template.sh
         done 
      done
   done
done
