#!/bin/bash
#SBATCH --job-name=tzsam_val
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/tzsam_%j.log
# TZ-SAM external forward-validation on LOTUS — decoupled from interactive ssh.
# Established framework: env_solar interpreter directly (no conda activate); PROJ_DATA; EE creds.
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | start $(date) ==="
$PY -u Solar-Siting/validate/validate_tzsam.py "$@"
echo "=== done $(date) ==="
