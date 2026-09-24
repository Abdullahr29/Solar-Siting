#!/bin/bash
#SBATCH --job-name=t3_score
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/users/ar29/t3_score.%j.out
# Train t-3 cap120 models + score t-2/t-3 locally on val embeddings. All local (no GEE).
cd /home/users/ar29/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
/home/users/ar29/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/t3/t3_train_score.py
