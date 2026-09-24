#!/bin/bash
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --job-name=paper_v3
#SBATCH --time=23:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/slurm_%j.out
echo "SLURM job $SLURM_JOB_ID on $(hostname) : $(nproc) cores visible, $(free -g | awk '/Mem:/{print $2}')GB"
bash /home/users/ar29/Solar_Workspace/Solar-Siting/run_all_v3.sh
