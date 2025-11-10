#!/usr/bin/env bash
# CNN (no embeddings) training launcher
# Example local run:
#   bash scripts/run_cnn_no_embeddings.sh --epochs 5 --dry_run
#
# For SLURM, uncomment and adjust:
# #SBATCH --job-name=cnn-no-emb
# #SBATCH --output=logs/cnn-no-emb-%j.out
# #SBATCH --error=logs/cnn-no-emb-%j.err
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G
# #SBATCH --time=12:00:00

set -euo pipefail

CONFIG="config.yaml"
OUTPUT_DIR="checkpoints"
EPOCHS=20
BATCH_SIZE=16
LR=0.0001
EMBED_DIM=128
FILTERS=128
DROPOUT=0.1
SEED=42
NUM_WORKERS=2
DEVICE=""
DRY_RUN=0
DATA_CSV=""
LABELS_CSV=""
SEQUENCES_CSV=""

print_help() {
  cat <<EOF
Run CNN (No Embeddings) Training

Flags:
  --config PATH            YAML config path (default: config.yaml)
  --output_dir DIR         Output checkpoints directory (default: checkpoints)
  --epochs N               Training epochs (default: 20)
  --batch_size N           Batch size (default: 16)
  --lr FLOAT               Learning rate (default: 1e-4)
  --embedding_dim N        Embedding dimension (default: 128)
  --num_filters N          CNN filters (default: 128)
  --dropout FLOAT          Dropout (default: 0.1)
  --seed N                 Random seed (default: 42)
  --num_workers N          DataLoader workers (default: 2)
  --device STR             Device override (e.g. cuda:0)
  --data_csv PATH          Single CSV containing seq+labels
  --labels_csv PATH        CSV with labels (and seq)
  --sequences_csv PATH     CSV with sequences (and labels)
  --dry_run                Build pipeline only, no training
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --embedding_dim) EMBED_DIM="$2"; shift 2;;
    --num_filters) FILTERS="$2"; shift 2;;
    --dropout) DROPOUT="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --num_workers) NUM_WORKERS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --data_csv) DATA_CSV="$2"; shift 2;;
    --labels_csv) LABELS_CSV="$2"; shift 2;;
    --sequences_csv) SEQUENCES_CSV="$2"; shift 2;;
    --dry_run) DRY_RUN=1; shift 1;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done

PY_ARGS=("--config" "$CONFIG" "--output_dir" "$OUTPUT_DIR" "--epochs" "$EPOCHS" "--batch_size" "$BATCH_SIZE" "--lr" "$LR" "--embedding_dim" "$EMBED_DIM" "--num_filters" "$FILTERS" "--dropout" "$DROPOUT" "--seed" "$SEED" "--num_workers" "$NUM_WORKERS")

if [[ -n "$DEVICE" ]]; then PY_ARGS+=("--device" "$DEVICE"); fi
if [[ -n "$DATA_CSV" ]]; then PY_ARGS+=("--data_csv" "$DATA_CSV"); fi
if [[ -n "$LABELS_CSV" ]]; then PY_ARGS+=("--labels_csv" "$LABELS_CSV"); fi
if [[ -n "$SEQUENCES_CSV" ]]; then PY_ARGS+=("--sequences_csv" "$SEQUENCES_CSV"); fi
if [[ "$DRY_RUN" -eq 1 ]]; then PY_ARGS+=("--dry_run"); fi

echo "[INFO] Launching CNN no-emb training: ${PY_ARGS[*]}"
python src/cnn_no_embeddings_train.py "${PY_ARGS[@]}"
