#!/bin/bash

#SBATCH --job-name=test_mlp
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mail-user=your_email@address.edu
#SBATCH --mail-type=END
#SBATCH --account=account_name
#SBATCH --output=test.out
#SBATCH --error=test.err

# Edit these before submitting
VERSION="Test_MLP"
DATA="data/20_25_500K.parquet"
BATCH_SIZE=256
# Set to "--compile" to enable torch.compile, or leave empty
COMPILE_FLAG=""

module purge
module load gcc/14.2.0
module load apptainer/1.3.4 pytorch/.2.9.0

cd "${SLURM_SUBMIT_DIR}"

apptainer run --nv "$CONTAINERDIR/pytorch-2.9.0.sif" test.py \
    --version "${VERSION}" \
    --data "${DATA}" \
    --batch-size "${BATCH_SIZE}" \
    ${COMPILE_FLAG}
