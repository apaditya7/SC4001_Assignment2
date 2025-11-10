#!/bin/bash
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P personal
#PBS -N cnn_no_embeddings
#PBS -j oe

cd "$PBS_O_WORKDIR"
mkdir -p "$PBS_O_WORKDIR/outputs"

# Load necessary modules (adjust based on your cluster)
module load miniforge3/25.3.1

# Activate your conda environment
conda activate myvenv

# Optionally select GPU
export CUDA_VISIBLE_DEVICES=0

echo "Job started at $(date) on $(hostname)"
echo "Python: $(which python)"
python --version

python src/cnn_no_embeddings_train.py \
    --data_csv /home/users/ntu/ktang022/scratch/SC4001_Assignment2/data/2018-06-06-ss.cleaned.csv \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir "$PBS_O_WORKDIR/outputs" \
    --num_filters 128 \
    --embedding_dim 128 \
    --dropout 0.1 \
    --num_workers 4 \
    > cnn_no_embeddings.out 2>&1

echo "Job completed at $(date)"