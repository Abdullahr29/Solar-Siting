#!/bin/bash
#SBATCH --job-name=rfsweep1
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-39%10
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/rfsweep1_%A_%a.log
# Stage 1: one GEE-feasible-frontier config per array task.
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
echo "=== node $(hostname) | task $SLURM_ARRAY_TASK_ID | start $(date) ==="
~/Solar_Workspace/envs/env_solar/bin/python -u Solar-Siting/archive/rf_sweep.py --config-id $SLURM_ARRAY_TASK_ID
echo "=== done $(date) ==="
