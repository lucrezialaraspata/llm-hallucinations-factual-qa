#!/bin/bash

#SBATCH -A IscrC_EXAM
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=123000
#SBATCH --job-name=kc-prepare-data
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_VISIBLE_DEVICES=0,1,2,3

source .venv/bin/activate

srun -u python -W ignore -m result_collector_kc
#srun -u accelerate launch --multi_gpu -m result_collector_kc