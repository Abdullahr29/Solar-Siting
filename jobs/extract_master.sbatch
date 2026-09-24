#!/bin/bash
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --job-name=extract_master
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/extract_master_%j.out
WS=/home/users/ar29/Solar_Workspace
echo "SLURM job $SLURM_JOB_ID on $(hostname): $(nproc) cores, $(free -g | awk "/Mem:/{print \$2}")GB, start $(date)"
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
$WS/envs/env_solar/bin/python $WS/Solar-Siting/pipeline/extract_pixels_master.py \
  --out $WS/Solar-Siting/artifacts/paper_v3/pools/px_master_train.npz \
  --cap-sol 0 --per-neg 8000 --procs 32 --split train
echo "done $(date)"
