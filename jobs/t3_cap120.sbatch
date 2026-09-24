#!/bin/bash
#SBATCH --job-name=t3_cap120
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/ar29/t3_cap120.%j.out
# Resumable t-3 (AEF install-3) sampling for the cap120 positive pool. Per-chip checkpoints;
# re-running skips done chips. yr==2017 reused free; yr>=2018 point-sampled from GEE.
cd /home/users/ar29/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/t3/t3_sample_pool.py --batch 2000
