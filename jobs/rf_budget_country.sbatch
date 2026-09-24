#!/bin/bash
#SBATCH --job-name=rfb_country
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/ar29/rfb_country.%j.out
# Stage 2: 13-country GEE forward-ROC for best-9MB-RF-per-pool + incumbent controls.
# Resumable via cached scores/country/*.npz, so timeouts are harmless.
cd /home/users/ar29/Solar_Workspace/Solar-Siting
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u validate/rf_budget_country.py
