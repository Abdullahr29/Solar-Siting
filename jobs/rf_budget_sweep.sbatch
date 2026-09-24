#!/bin/bash
#SBATCH --job-name=rfbudget
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/home/users/ar29/rfbudget_%x_%j.out
# POOL is passed via --export (e.g. sbatch --export=ALL,POOL=T6000000 ...)
export NJOBS=12
export BUDGET_LEAVES=63000   # ~8.8 MB at ~140 B/leaf worst case -> near the 9 MB ceiling
cd /home/users/ar29/Solar_Workspace/Solar-Siting
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u train/rf_budget_sweep.py
