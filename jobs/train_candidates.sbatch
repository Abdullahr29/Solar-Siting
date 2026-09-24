#!/bin/bash
#SBATCH --job-name=train_cand
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/ar29/train_cand.%j.out
export NJOBS=12
export TS=10000000,6000000,4000000,1000000
cd /home/users/ar29/Solar_Workspace/Solar-Siting
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u train/train_candidates.py
