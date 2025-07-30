#!/bin/bash

#SBATCH -A [removed for anonymization]
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --mem=123000
#SBATCH --job-name=kc-prepare-data
#SBATCH --out=output_kc.log
#SBATCH --err=error_kc.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mail@domain.com

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

source .venv/bin/activate

srun -u python -m result_collector_kc
#srun -u accelerate launch --multi_gpu -m result_collector_kc