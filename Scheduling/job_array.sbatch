#!/bin/bash
#SBATCH --job-name=jobarray
#SBATCH --output=jobarray_%A_%a.out
#SBATCH --error=jobarray_%A_%a.err
#SBATCH --array=1-6
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000

module load python
# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Do some work based on the SLURM_ARRAY_TASK_ID
python my_script.py $SLURM_ARRAY_TASK_ID

