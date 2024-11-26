#!/bin/bash
#SBATCH --job-name=map
#SBATCH --output=logs-mapper/test_%j.log
#SBATCH --error=logs-mapper/test_%j.err
#SBATCH --nodes=1
#SBATCH --exclude=g0001
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu
#SBATCH --export=ALL


srun slurm-train.sh
echo "Done with job $SLURM_JOB_ID"