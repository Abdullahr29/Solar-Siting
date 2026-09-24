#!/bin/bash
#SBATCH --account=gbov
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --job-name=px_census
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=/home/users/ar29/Solar_Workspace/Solar-Siting/artifacts/stats/census_s0_%j.out
WS=/home/users/ar29/Solar_Workspace
echo "SLURM job $SLURM_JOB_ID on $(hostname): $(nproc) cores, $(free -g | awk "/Mem:/{print \$2}")GB"
$WS/envs/env_solar/bin/python $WS/Solar-Siting/pixel_census_v3.py --procs 32 --finite-sample 2000
