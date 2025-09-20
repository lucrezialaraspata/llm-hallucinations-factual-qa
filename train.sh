#!/bin/bash

#SBATCH -A IscrC_EXAM
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=123000
#SBATCH --job-name=train
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

conda activate hallucination

module load python

srun -u python -W ignore -m classifier_model