#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P personal
#PBS -N transformer_cleaned
#PBS -j oe

# Change to the directory where the job was submitted
cd "$PBS_O_WORKDIR"

# Create output directory
mkdir -p "$PBS_O_WORKDIR/outputs"

# Load necessary modules (adjust based on your cluster)
module load miniforge3/25.3.1

# Activate your conda environment
conda activate myenv

# Set CUDA visible devices (optional, if you want to control GPU usage)
export CUDA_VISIBLE_DEVICES=0

# Print some runtime info
echo "Job started at $(date) on $(hostname)"

echo "Python: $(which python)"
python --version

# Run the training script (mapped to this repo's CLI)
python src/transformer_train_clean.py \
    --data_csv /home/users/ntu/ktang022/scratch/SC4001_Assignment2/data/2018-06-06-ss.cleaned.csv \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir "$PBS_O_WORKDIR/outputs" \
    --num_heads 8 \
    --num_layers 4 \
    --ff_dim 512 \
    --dropout 0.1 \
    --num_workers 4 \
    > transformer_cleaned.out 2>&1

echo "Job completed at $(date)"