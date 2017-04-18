#!/bin/bash 
#SBATCH --job-name=example_gpu_sbatch
#SBATCH --output=example_gpu_sbatch_%j.out
#SBATCH --error=example_gpu_sbatch_%j.err
#SBATCH --account=you_pi_account
#SBATCH --time=00:05:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email_address
#enter gpu2 if you want to use GPU nodes from Midway2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

module load python
#cd your_working_directory
echo “job started at `date`”

python my_script.py

echo “job finished at `date`”

