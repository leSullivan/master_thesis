#!/bin/bash
model_name="cyclegan"
lambda_cycle="0"
# g_type="resnet-6"
# d_type="patch"
lambda_perceptual="5"
lambda_gan="0.5"
lambda_cycle="10"


for g_type in resnet-6 unet_256; do 
   for d_type in patch vagan; do
         sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan slurm_template.sh
   done
done
