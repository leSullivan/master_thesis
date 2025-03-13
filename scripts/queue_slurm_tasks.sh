#!/bin/bash
model_name="cgan"
g_type="unet_128"
d_type="vagan"
lambda_perceptual="5"
lambda_gan="0.5"
lambda_cycle="10"
ngf=64
crop=False
img_h=512
img_w=768

for model_name in cyclegan; do 
  for g_type in unet_128 unet_256; do
    for d_type in vagan patch; do
      for ngf in 128; do
        for lambda_perceptual in 1 5 10; do
          for lambda_cycle in 1 3 5; do
            sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop,IMG_H=$img_h,IMG_W=$img_w slurm_template.sh
          done
        done
      done
    done
  done
done


