#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P personal
#PBS -N esm2_cnn
#PBS -j oe

# Change to the directory where the job was submitted
cd "$PBS_O_WORKDIR"

# Create output directory
mkdir -p "$PBS_O_WORKDIR/outputs"

# Load necessary modules (adjust based on your cluster)
module load miniforge3/25.3.1

# Activate your conda environment
conda activate myvenv

# Set CUDA visible devices (optional, if you want to control GPU usage)
export CUDA_VISIBLE_DEVICES=0

# Optional: echo GPU info if available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU status:"; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
fi

# Print some runtime info
echo "Job started at $(date) on $(hostname)"

echo "Python: $(which python)"
python --version

# Run the training script (ESM2+CNN)
python src/esm2_cnn_train.py \
    --sequences_csv /home/users/ntu/ktang022/scratch/SC4001_Assignment2/data/2018-06-06-pdb-intersect-pisces.csv \
    --labels_csv /home/users/ntu/ktang022/scratch/SC4001_Assignment2/data/2018-06-06-ss.cleaned.csv \
    --id_column pdb_id \
    --batch_size 16 \
    --epochs 100 \
    --lr 2e-4 \
    --output_dir "$PBS_O_WORKDIR/outputs" \
    --num_filters 128 \
    --dropout 0.1 \
    --esm_model_name esm2_t30_150M_UR50D \
    --esm_batch_size 8 \
    --num_workers 4 \
    1> esm2_cnn.out 2>&1

echo "Job completed at $(date)"