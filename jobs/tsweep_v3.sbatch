#!/bin/bash
#SBATCH --job-name=tsweep_v3
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/ar29/tsweep_v3.%j.out
export NJOBS=12         # lower parallelism -> caps transient per-tree (weight/count) memory
export N_NEG=0          # 0 -> ALL 64.65M negatives, fixed & identical for every T (ratio floats)
cd /home/users/ar29/Solar_Workspace/Solar-Siting
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u archive/tsweep_v3.py
