#!/bin/bash
model_name="turbo_cyclegan"
d_type="vagan"
lambda_perceptual="5"
lambda_gan="0.5"
lambda_cycle="10"
ngf=64
crop=False

for lambda_cycle in 1 5 10; do
    for lambda_perceptual in 1 5 10; do
        sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_multigpu_template.sh
    done
done


