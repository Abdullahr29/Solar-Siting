#!/bin/bash
#SBATCH --job-name=cand_country
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/ar29/cand_country.%j.out
cd /home/users/ar29/Solar_Workspace/Solar-Siting
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u validate/country_candidates.py
