!/bin/bash
for model_name in cyclegan; do
   for g_type in resnet-6 resnet-9; do 
      for lambda_cycle in 0.1 1 10 20; do 
         sbatch --export=MODEL_NAME=$model_name,LAMBDA_CYCLE=$lambda_cycle,G_TYPE=$g_type,D_TYPE=vagan, slurm_template.sh
      done
   done
done
