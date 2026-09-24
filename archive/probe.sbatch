#!/bin/bash
#SBATCH --partition=standard
#SBATCH --job-name=p3probe
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/users/ar29/p3probe.out
bash /home/users/ar29/lotus_probe.sh
