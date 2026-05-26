#!/bin/bash
#SBATCH --output=/scratch/negishi/mei75/other-chud-projects/gw/ml_pipeline/outputs/outputs-5-27-26-t1
#SBATCH --partition=cpu
#SBATCH --job-name='cluster'
#SBATCH --account=meji
#SBATCH --qos=standby
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=7500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meitriton+HPC@gmail.com

# --- Change to working directory ---
cd /scratch/negishi/mei75/other-chud-projects/gw/ml_pipeline
echo 'Working directory: /scratch/negishi/mei75/other-chud-projects/gw/ml_pipeline'

# --- Set Environment ---
module purge

module load conda
module load use.own
module load conda-env/py-gnn-env-py3.12.11

python -c "import torch._C"

# --- Run the Python script ---
python /scratch/negishi/mei75/other-chud-projects/gw/ml_pipeline/_06_training.py