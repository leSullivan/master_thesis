#!/bin/bash
# for g_type in resnet-6 unet; do
#   for d_type in basic patch pixel; do
    # sbatch --export=D_TYPE=$d_type,G_TYPE=$g_type slurm_template.sh
#   done
# done

sbatch --export=D_TYPE=basic,G_TYPE=resnet-6 slurm_template.sh