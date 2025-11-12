# Notebook Organization Summary

## Overview
This directory contains 10 main notebooks for protein secondary structure prediction, organized into 5 model architectures with 2 variants each (with/without ESM2 embeddings).

## Active Notebooks (10 total)

### 1. CNN Models
- **`1_cnn_esm2.ipynb`** ✅ - CNN with ESM2 pre-trained embeddings
  - Uses 640-dim ESM2 features directly in conv layers
  
- **`1_cnn_no_embeddings.ipynb`** ✅ - CNN with learnable embeddings
  - Learns amino acid embeddings from scratch
  - Includes positional encoding

### 2. Transformer Models
- **`2_transformer_esm2.ipynb`** ✅ - Transformer with ESM2 embeddings
  - ESM2 features replace learnable embedding layer
  
- **`2_transformer_no_embeddings.ipynb`** ✅ - Transformer with learnable embeddings
  - Standard transformer architecture

### 3. CNN-Transformer Hybrid Models
- **`3_cnn_transformer_esm2.ipynb`** ✅ - Hybrid with ESM2
  - ESM2 → CNN (local patterns) → Transformer (global dependencies)
  
- **`3_cnn_transformer_no_embeddings.ipynb`** ✅ - Hybrid without ESM2
  - Learnable embeddings → CNN → Transformer

### 4. BiLSTM Models
- **`4_bilstm_esm2.ipynb`** ✅ - BiLSTM with ESM2 embeddings
  - Bidirectional LSTM on pre-trained features
  
- **`4_bilstm_no_embeddings.ipynb`** ✅ - BiLSTM with learnable embeddings
  - Learns sequence representations during training

### 5. BiLSTM-CNN Hybrid Models
- **`5_bilstm_cnn_esm2.ipynb`** ✅ - BiLSTM-CNN hybrid with ESM2
  - ESM2 → BiLSTM (contextual) → CNN (local motifs)
  
- **`5_bilstm_cnn_no_embeddings.ipynb`** ✅ - BiLSTM-CNN hybrid without ESM2
  - Learnable embeddings → BiLSTM → CNN

## Implementation Status
✅ **All 10 notebooks are complete!** Each notebook contains full implementations ready for training and evaluation.

## Archive Directory
The `archive/` folder contains experimental notebooks that are not part of the main 10-model structure:
- **RNN-based models** (2 files): `rnn_esm2.ipynb`, `rnnNoEmbeddings.ipynb` - Single RNN models
- **RNN-BiLSTM ensembles** (2 files): `ensemble_rnn_bilstm_esm2.ipynb`, `ensemble_rnn_bilstm_noemb.ipynb`
- **Old transformer variant** (1 file): `transfomerTrainSeq.ipynb`

These are kept for reference but not part of the core model comparison.

## Naming Convention
- Number (1-5): Model architecture type
- Architecture name: `cnn`, `transformer`, `cnn_transformer`, `bilstm`, `bilstm_cnn`
- Variant: `_esm2` for pre-trained embeddings, `_no_embeddings` for learnable

## Model Descriptions

### ESM2 Embeddings (Pre-trained)
All models with `_esm2` suffix use Facebook's ESM2 protein language model (640-dim embeddings per residue). These provide:
- Rich biochemical context
- Better generalization with less data
- Pretrained features that capture evolutionary information

### Learnable Embeddings (No ESM2)
All models with `_no_embeddings` suffix learn embeddings from scratch:
- Standard amino acid vocabulary encoding
- Embedding layer trained end-to-end
- More parameters to learn but greater flexibility

## Data
All notebooks use:
- **Input**: `data/2018-06-06-pdb-intersect-pisces.csv` (sequences)
- **Labels**: `data/2018-06-06-ss.cleaned.csv` (8-class and 3-class secondary structure)

## Output
Each notebook produces:
- **Q8 predictions**: 8-class secondary structure (H, G, I, E, B, T, S, C)
- **Q3 predictions**: 3-class secondary structure (Helix, Sheet, Coil)
- **Metrics**: Accuracy, per-class metrics, confusion matrices
- **Checkpoints**: Best model weights saved during training
