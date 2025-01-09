#!/bin/bash
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu-debug
#SBATCH --mem=25000M
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --output=testGPU.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."
module load scicomp-python-env
#export OMP_PROC_BIND=true
#module load nvidia-pytorch/20.02-py3

# For the next line to work, you need to be in the
# hpc-examples directory.
python goodConfigGPU_real_d_grid.py #goodConfigGPU.py 