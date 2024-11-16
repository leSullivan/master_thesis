#!/bin/bash
model_name="turbo_cyclegan"
g_type="unet_128"
d_type="vagan"
lambda_perceptual="5"
lambda_gan="0.5"
lambda_cycle="10"
ngf=64
crop=False

# next experiments

# crop false
# cycle gan with best results in cropped and not cropped
# get turbo turbo cycle gan working

# for model_name in cgan; do 
#   for g_type in unet_128 unet_256; do
#     for d_type in patch vagan; do
#       for ngf in 64 128; do
#         for lambda_perceptual in 5 10 15; do
#           if [ $model_name == "turbo_cyclegan" ]; then
#             sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_multigpu_template.sh
#           else
#             sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_template.sh
#           fi
#         done
#       done
#     done
#   done
# done


# for model_name in cyclegan; do 
#   # for g_type in resnet-6 resnet-skip-con; do
#   for g_type in unet_128 unet_256; do
#     for d_type in patch vagan; do
#       for ngf in 64 128; do
#         for lambda_perceptual in 10 15; do
#          for lambda_cycle in 5 10 15; do
#           if [ $model_name == "turbo_cyclegan" ]; then
#             sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_multigpu_template.sh
#           else
#             sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_template.sh
#           fi
#           done
#         done
#       done
#     done
#   done
# done

# sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_multigpu_template.sh

for crop in True; do
  for model_name in cgan; do 
    for g_type in resnet-6; do
      for d_type in patch vagan; do
        for ngf in 32 64 128; do
            if [ $model_name == "turbo_cyclegan" ]; then
              sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_multigpu_template.sh
            else
              sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=$d_type,LAMBDA_PERCEPTUAL=$lambda_perceptual,LAMBDA_GAN=$lambda_gan,NGF=$ngf,CROP=$crop slurm_template.sh
            fi
        done
      done
    done
  done
done