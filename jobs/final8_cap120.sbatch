#!/bin/bash
#SBATCH --job-name=f8_cap120
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/users/ar29/f8_cap120.%j.out
cd /home/users/ar29/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/validate/final8_ladder.py --group cap120
