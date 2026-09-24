#!/bin/bash
#SBATCH --job-name=mcda_rast
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/mcda_rast_%j.log
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
export WS=~/Solar_Workspace
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
echo "=== node $(hostname) | $* | start $(date) ==="
~/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/figures/mcda_raster.py "$@"
echo "=== done $(date) ==="
