#!/bin/bash
#SBATCH --job-name=rfsweep2
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/rfsweep2_%j.log
# Stage 2: honest country-forward validation of the shortlist (GEE, sequential, resumable).
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
PY=~/Solar_Workspace/envs/env_solar/bin/python
echo "=== node $(hostname) | start $(date) ==="
$PY -u Solar-Siting/archive/rf_sweep_stage2.py
echo "----- aggregating -----"
$PY -u Solar-Siting/archive/rf_sweep_stage2_aggregate.py
echo "=== done $(date) ==="
