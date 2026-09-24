#!/bin/bash
#SBATCH --job-name=treecount
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/home/users/ar29/treecount.%j.out
# Probe n_estimators {15,20,25,30} at 9MB budget on POOL (default cap120). All-local scoring.
cd /home/users/ar29/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
export POOL=${POOL:-px_v3}
export BUDGET_LEAVES=63000
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/train/treecount_sweep.py
