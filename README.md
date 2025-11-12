# SC4001 Assignment 2 – Protein Secondary Structure Prediction

This repository contains notebooks and Python modules for training deep learning models for protein secondary structure prediction, using both pre-trained ESM2 embeddings and learnable embeddings.

## Repository Structure

### Notebooks Overview

| # | Model Architecture | With ESM2 | Without ESM2 |
|---|-------------------|-----------|--------------|
| 1 | **CNN** | `1_cnn_esm2.ipynb` | `1_cnn_no_embeddings.ipynb` |
| 2 | **Transformer** | `2_transformer_esm2.ipynb` | `2_transformer_no_embeddings.ipynb` |
| 3 | **CNN-Transformer Hybrid** | `3_cnn_transformer_esm2.ipynb` | `3_cnn_transformer_no_embeddings.ipynb` |
| 4 | **BiLSTM** | `4_bilstm_esm2.ipynb` | `4_bilstm_no_embeddings.ipynb` |
| 5 | **BiLSTM-CNN Hybrid** | `5_bilstm_cnn_esm2.ipynb` | `5_bilstm_cnn_no_embeddings.ipynb` |

### Notebooks

The repository contains 10 main notebooks organized by model architecture:

#### Models with ESM2 Pre-trained Embeddings
1. **`1_cnn_esm2.ipynb`** - CNN with ESM2 embeddings
   - ESM2 embeddings fed directly into 1D convolutional layers
   - Operates on semantically enriched residue representations (640-dim)
   
2. **`2_transformer_esm2.ipynb`** - Transformer with ESM2 embeddings
   - ESM2 embeddings serve as initial token representations
   - Replaces learnable embedding layer with pre-trained features
   
3. **`3_cnn_transformer_esm2.ipynb`** - CNN-Transformer hybrid with ESM2
   - ESM2 embeddings → CNN for local patterns → Transformer for global dependencies
   
4. **`4_bilstm_esm2.ipynb`** - BiLSTM with ESM2 embeddings
   - BiLSTM models residue dependencies bidirectionally
   - Pre-trained features enable better generalization
   
5. **`5_bilstm_cnn_esm2.ipynb`** - BiLSTM-CNN hybrid with ESM2
   - BiLSTM for contextual understanding → CNN for local motif detection
   - Pre-trained embeddings enhance both stages

#### Models with Learnable Embeddings (No ESM2)
1. **`1_cnn_no_embeddings.ipynb`** - CNN with learnable embeddings
   - Learns amino acid embeddings from scratch
   - Includes positional encoding + convolutional stack
   
2. **`2_transformer_no_embeddings.ipynb`** - Transformer with learnable embeddings
   - Standard Transformer with learned amino acid representations
   
3. **`3_cnn_transformer_no_embeddings.ipynb`** - CNN-Transformer hybrid
   - Learnable embeddings → CNN → Transformer
   
4. **`4_bilstm_no_embeddings.ipynb`** - BiLSTM with learnable embeddings
   - Learns sequence representations during training
   
5. **`5_bilstm_cnn_no_embeddings.ipynb`** - BiLSTM-CNN hybrid
   - Learnable embeddings → BiLSTM → CNN

**Note:** Older experimental notebooks (ensembles, duplicates, etc.) have been moved to `notebooks/archive/` for reference.

### Python Modules and Scripts

#### ESM2-based Training
- `src/esm2_cnn_train.py`: CNN model with ESM2 embeddings + CLI interface
- `scripts/run_esm2_cnn.sh` & `scripts/pbs_run_esm2_cnn.sh`: HPC launcher scripts for ESM2-CNN

#### Transformer Training
- `src/transformer_train_clean.py`: Transformer model with learnable embeddings + CLI interface
- `scripts/run_transformer_clean.sh` & `scripts/pbs_run_transformer_clean.sh`: HPC launchers for Transformer

#### CNN (No Embeddings) Training
- `src/cnn_no_embeddings_train.py`: CNN with learnable embeddings + CLI interface
- `scripts/run_cnn_no_embeddings.sh`: HPC launcher for CNN

### Data
- `data/2018-06-06-pdb-intersect-pisces.csv`: Protein sequence data
- `data/2018-06-06-ss.cleaned.csv`: Secondary structure labels (8-class and 3-class)

## Quick start

1) Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Dry-run to verify wiring (no training, no GPU required):

```bash
python src/transformer_train_clean.py --dry_run
```

3) Full Transformer training (defaults read from `config.yaml` when present):

```bash
python src/transformer_train_clean.py \
  --config config.yaml \
  --output_dir checkpoints \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4
```

The script auto-discovers a CSV with `seq`, `sst8`, and `sst3` columns. By default it uses `data/2018-06-06-ss.cleaned.csv`. You can override with `--data_csv PATH` (or `--labels_csv/--sequences_csv`).

Artifacts:
- Best model: `checkpoints/best_transformer_clean.pt`
- Metrics JSON: `checkpoints/metrics_transformer_clean.json`

## Transformer HPC usage (SLURM)

Edit `scripts/run_transformer_clean.sh` to match your cluster (uncomment and modify `#SBATCH` lines, modules, and env activation). Then submit:

```bash
sbatch scripts/run_transformer_clean.sh --epochs 40 --batch_size 32
```

Alternatively, run interactively on a login or compute node:

```bash
bash scripts/run_transformer_clean.sh --dry_run
```

Transformer common flags:
- `--config config.yaml` – read defaults from YAML
- `--output_dir DIR` – where to write checkpoints/metrics
- `--epochs`, `--batch_size`, `--lr` – training hyperparams
- `--embedding_dim`, `--layers`, `--heads`, `--ff_dim`, `--dropout` – model hyperparams
- `--device cuda:0` – force a device (auto by default)
- `--data_csv PATH` – single CSV containing `seq`, `sst8`, `sst3`

### Transformer Notes
## CNN (No Embeddings) Quick start

Dry-run:
```bash
python src/cnn_no_embeddings_train.py --dry_run
```

Full training:
```bash
python src/cnn_no_embeddings_train.py \
  --config config.yaml \
  --output_dir checkpoints \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_filters 128
```

Artifacts:
- Best model: `checkpoints/best_cnn_no_embeddings.pt`
- Metrics JSON: `checkpoints/metrics_cnn_no_embeddings.json`

### CNN HPC usage (PBS example)
Submit:
```bash
qsub scripts/pbs_run_transformer_clean.sh  # transformer example
```
For CNN with SLURM style (adapt lines in `run_cnn_no_embeddings.sh`):
```bash
bash scripts/run_cnn_no_embeddings.sh --epochs 40 --num_filters 256
```

CNN common flags:
- `--num_filters` – channels in conv stack (defaults to 128 or `hidden_dim` from YAML)
- `--embedding_dim`, `--dropout`, plus all training flags identical to transformer script

### CNN Notes
- Implements per-residue Q8/Q3 classification and SOV-Q3 metric.
- Reuses the same CSV and splitting logic as the transformer.
- Best checkpoint selected by validation Q8 accuracy.


- The refactor follows the original notebook logic closely: tokenized sequences, positional encoding, Transformer encoder, dual heads for Q8 and Q3, best checkpoint by Val Q8.
- The YAML in `config.yaml` was authored for RNN/LSTM originally; this script reuses compatible fields (e.g., `embedding_dim`, `layers`, `dropout`, training settings). Transformer-specific params (`num_heads`, `ff_dim`) can be passed via CLI.
- Set `--dry_run` to validate data loading and pipeline construction without any training.
