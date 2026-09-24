#!/bin/bash
#SBATCH --job-name=mcda_v3
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/paper_v3/logs/mcda_v3_slurm_%j.log
# MCDA v3 benchmark on LOTUS — decoupled from interactive ssh (connection-robust).
# Established framework: PROJ_DATA for GDAL/geopandas local-UTM reprojection; env_solar; EE creds.
export PROJ_DATA=~/Solar_Workspace/envs/env_solar/share/proj
cd ~/Solar_Workspace
export PYTHONPATH="/gws/ssde/j25b/gbov/abdullah_solar/Solar-Siting:$PYTHONPATH"
echo "=== node $(hostname) | start $(date) ==="
bash Solar-Siting/jobs/run_mcda_v3.sh
echo "=== done $(date) ==="
