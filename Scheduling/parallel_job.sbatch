#!/bin/sh
#SBATCH --time=00:05:00
#SBATCH --ntasks=4

module load python
module load parallel

srun="srun --exclusive -N1 -n1"

parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog runtask.log --resume"

$parallel "$srun python my_script.py {1} > runtask.{1}" ::: {1..16}
