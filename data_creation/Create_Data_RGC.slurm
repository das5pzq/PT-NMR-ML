#!/bin/bash
#SBATCH -p parition_name
#SBATCH --job-name=rgc_data_generation
#SBATCH --output=job_logs/rgc_job_%A_%a.out
#SBATCH --error=job_logs/rgc_job_%A_%a.err
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -A account_name
#SBATCH --array=1-200     # Parallel jobs; total samples = NUM_SAMPLES * array_size

set -euo pipefail

# ===== Global Configuration =====
# Submit from the RGC project directory (scripts live here).
OUTPUT_DIR="Training_Data_RGC"
OUTPUT_FILENAME="Training_Data_RGC_200K.parquet"
NUM_SAMPLES=1000                       # Samples per task; TOTAL = NUM_SAMPLES * SLURM_ARRAY_TASK_MAX
ADD_NOISE=1                           # 0=no noise, 1=add noise
NOISE_LEVEL=0.000027                  # Gaussian noise std (RGC wing scale)

mkdir -p job_logs

echo "Starting RGC job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Working dir: $(pwd)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Output filename: ${OUTPUT_FILENAME}"
echo "Samples per job: ${NUM_SAMPLES}"
echo "Add noise: ${ADD_NOISE}"
echo "Noise level: ${NOISE_LEVEL}"


mkdir -p "${OUTPUT_DIR}"
chmod 755 "${OUTPUT_DIR}"

echo "Generating RGC signal data..."
python Create_Training_Data_RGC.py \
    --job_id "${SLURM_ARRAY_TASK_ID}" \
    --num_samples "${NUM_SAMPLES}" \
    --add_noise "${ADD_NOISE}" \
    --noise_level "${NOISE_LEVEL}" \
    --output_dir "${OUTPUT_DIR}"

echo "Signal generation complete for task ${SLURM_ARRAY_TASK_ID}"

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

if [ "${SLURM_ARRAY_TASK_ID}" -eq "1" ]; then
    echo "Submitting merge job after array completion..."
    MERGE_SCRIPT="merge_job_rgc.sh"
    cat > "${MERGE_SCRIPT}" << EOF
#!/bin/bash
#SBATCH -p standard
#SBATCH --job-name=merge_rgc_parquet
#SBATCH --output=job_logs/merge_rgc_%j.out
#SBATCH --error=job_logs/merge_rgc_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -A localhost

set -euo pipefail

python merge.py "${OUTPUT_DIR}" "${OUTPUT_FILENAME}" "Sample_*.parquet"
EOF

    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} "${MERGE_SCRIPT}"
    echo "Merge job submitted with dependency on array job ${SLURM_ARRAY_JOB_ID}"
fi

echo "Job ${SLURM_JOB_ID} complete at $(date)"
