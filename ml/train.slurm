#!/bin/bash

#SBATCH --time=72:00:00  
#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=16   
#SBATCH --partition=gpu   # partition
#SBATCH --gres=gpu:a6000:1  # just leave 'gpu' to get most available GPU
#SBATCH --mail-user=your_email@address.edu
#SBATCH --mail-type=END
#SBATCH --account=account_name   
#SBATCH --output=training.out
#SBATCH --error=training.err

module purge 
module load gcc/14.2.0
module load apptainer/1.3.4 pytorch/.2.9.0

apptainer run --nv $CONTAINERDIR/pytorch-2.9.0.sif sample_script.py