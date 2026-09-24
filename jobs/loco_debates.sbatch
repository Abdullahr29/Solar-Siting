#!/bin/bash
#SBATCH --job-name=loco_deb
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/ar29/loco_deb.%j.out
# LOCO for 3 debates (12 retrains, resumable). 256G for the 40GB master 6M rebuilds.
cd /home/users/ar29/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/validate/loco_debates.py
