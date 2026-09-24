#!/bin/bash
#SBATCH --job-name=global_suit
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/global_suit_%j.log
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | start $(date) ==="
$PY -u Solar-Siting/analysis/global_suitable.py
echo "=== done $(date) ==="
