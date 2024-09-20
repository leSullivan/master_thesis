#!/bin/bash
 for model_name in cgan cyclegan turbo_cyclegan; do 
    sbatch --export=MODEL_NAME=$model_name slurm_template.sh
 done
