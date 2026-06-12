#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=data_generation
#SBATCH --output=job_logs/job_%A_%a.out  # Redirect output to job_logs subdirectory
#SBATCH --error=job_logs/job_%A_%a.err   # Redirect error to job_logs subdirectory
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -A account_name
#SBATCH --array=1-1     # Parallel jobs; total samples = NUM_SAMPLES * array_size

# ===== Global Configuration =====
# Set these variables to configure the job
PROJECT_DIR="`/path/to/project`"  ### Change this to your own directory
OUTPUT_DIR="${PROJECT_DIR}/Test"
OUTPUT_FILENAME="Test.parquet"  # Final merged filename
MODE="deuteron"                       # Options: deuteron, proton
POLARIZATION_TYPE="vector"            # Options: vector, tensor
TENSOR_DOMAIN="None"                  # Options: None, phase, time (tensor only)
NUM_SAMPLES=100                       # Samples per task; TOTAL = NUM_SAMPLES * SLURM_ARRAY_TASK_MAX
ADD_NOISE=0                           # 0=no noise, 1=add noise
OVERSAMPLING=0                        # 0=no oversampling, 1=enable oversampling
OVERSAMPLED_VALUE=0.0005              # Value to oversample around
OVERSAMPLING_UPPER_BOUND=0.008       # Upper bound for oversampling range
OVERSAMPLING_LOWER_BOUND=0.0002       # Lower bound for oversampling range
UPPER_BOUND=0.005                       # Upper bound for P value (not oversampled)
LOWER_BOUND=0.0001                       # Lower bound for P value (not oversampled)
P_MAX=0.02                             # Maximum polarization value
ALPHA=0.3                             # Decay rate for power law distribution
BASELINE=0                         # 0=no baseline, 1=add baseline
NOISE_LEVEL=0.000002                  # Standard deviation of Gaussian noise
SHIFTING=1                         # 0=no shifting, 1=enable shifting
BOUND=0.08                            # Bound for shifting

mkdir -p job_logs

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration:"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Output filename: ${OUTPUT_FILENAME}"
echo "  - Mode: ${MODE}"
echo "  - Polarization type: ${POLARIZATION_TYPE}"
echo "  - Tensor domain: ${TENSOR_DOMAIN}"
echo "  - Samples per job: ${NUM_SAMPLES}"
echo "  - Add noise: ${ADD_NOISE}"
echo "  - Oversampling: ${OVERSAMPLING}"
echo "  - Oversampled value: ${OVERSAMPLED_VALUE}"
echo "  - Oversampling upper bound: ${OVERSAMPLING_UPPER_BOUND}"
echo "  - Oversampling lower bound: ${OVERSAMPLING_LOWER_BOUND}"
echo "  - Upper bound: ${UPPER_BOUND}"
echo "  - Lower bound: ${LOWER_BOUND}"
echo "  - P_max: ${P_MAX}"
echo "  - Alpha: ${ALPHA}"
echo "  - Baseline: ${BASELINE}"
echo "  - Noise level: ${NOISE_LEVEL}"
echo "  - Shifting: ${SHIFTING}"
echo "  - Bound: ${BOUND}"
module purge
echo "Modules purged"
echo "Modules loaded"
module load miniforge/24.11.3-py3.12
mkdir -p ${OUTPUT_DIR}
chmod 755 ${OUTPUT_DIR}
echo "Output directory created: ${OUTPUT_DIR}"


# ===== Part 1: Generate Signal Data =====
echo "Generating signal data..."

echo "Running signal generation script..."
python Create_Training_Data.py \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --mode ${MODE} \
    --polarization_type ${POLARIZATION_TYPE} \
    --tensor-domain ${TENSOR_DOMAIN} \
    --num_samples ${NUM_SAMPLES} \
    --add_noise ${ADD_NOISE} \
    --oversampling ${OVERSAMPLING} \
    --oversampled_value ${OVERSAMPLED_VALUE} \
    --oversampling_upper_bound ${OVERSAMPLING_UPPER_BOUND} \
    --oversampling_lower_bound ${OVERSAMPLING_LOWER_BOUND} \
    --upper_bound ${UPPER_BOUND} \
    --lower_bound ${LOWER_BOUND} \
    --p_max ${P_MAX} \
    --alpha ${ALPHA} \
    --baseline ${BASELINE} \
    --noise_level ${NOISE_LEVEL} \
    --shifting ${SHIFTING} \
    --bound ${BOUND} \
    --output_dir ${OUTPUT_DIR}

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Signal generation failed. Exiting."
    exit 1
fi

echo "Signal generation complete for task ${SLURM_ARRAY_TASK_ID}"

# ===== Part 2: Create the merge.py script directly =====
echo "Creating the merge.py script..."

cat > merge.py << 'EOF'
"""
Stream-merge parquet shards using pyarrow to avoid high memory.
"""
import sys
import glob
from pathlib import Path
import pyarrow.dataset as ds
import pyarrow.parquet as pq

def main():
    output_dir = Path(sys.argv[1])
    output_filename = Path(sys.argv[2])
    pattern = sys.argv[3] if len(sys.argv) > 3 else "Sample_*.parquet"

    # Expand glob pattern to find matching files
    glob_pattern = str(output_dir / pattern)
    matching_files = glob.glob(glob_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: {glob_pattern}")
    
    print(f"Merging {len(matching_files)} files in {output_dir} matching {pattern} -> {output_filename}")
    dataset = ds.dataset(matching_files, format="parquet")
    scanner = dataset.scanner()

    writer = None
    total_rows = 0
    for batch in scanner.to_batches():
        if writer is None:
            writer = pq.ParquetWriter(output_filename, batch.schema, compression="snappy")
        writer.write_batch(batch)
        total_rows += batch.num_rows
    if writer:
        writer.close()
    print(f"Done. Files merged: {len(dataset.files)}, rows: {total_rows:,}")

if __name__ == "__main__":
    main()
EOF

# ===== Part 3: Submit merge job once per array (task 1), after array finishes =====
if [ "${SLURM_ARRAY_TASK_ID}" -eq "1" ]; then
    echo "Submitting merge job after array completion..."
    MERGE_SCRIPT="merge_job.sh"
    cat > ${MERGE_SCRIPT} << EOF
#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=merge_parquet
#SBATCH --output=job_logs/merge_%j.out
#SBATCH --error=job_logs/merge_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -A spinquest_standard

set -euo pipefail

module purge
module load miniforge/24.11.3-py3.12

cd "${PROJECT_DIR}"
python merge.py "${OUTPUT_DIR}" "${OUTPUT_FILENAME}" "Sample_*.parquet"
EOF

    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} ${MERGE_SCRIPT}
    echo "Merge job submitted with dependency on array job ${SLURM_ARRAY_JOB_ID}"
fi

echo "Job ${SLURM_JOB_ID} complete at $(date)"