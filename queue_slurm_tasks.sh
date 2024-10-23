#!/bin/bash
model_name="turbo_cyclegan"
lambda_cycle="0"
g_type="stan_unet_6_layer"
# d_type="vagan"
lambda_perceptual="5"
lambda_gan="0.5"
lambda_cycle="10"
ngf=64



for model_name in cgan; do 
  for g_type in stan_unet_6_layer unet_128 unet_256 resnet-9; do
    for d_type in patch vagan; do
      for ngf in 32 128; do
        sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf slurm_template.sh
      done
    done
  done
done

