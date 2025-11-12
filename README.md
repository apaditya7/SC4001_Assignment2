# Protein Secondary Structure Prediction

Deep learning models for predicting protein secondary structure (Q3/Q8 classification) from amino acid sequences.

## Overview

This project predicts local structural elements (helices, strands, loops) directly from protein sequences, replacing expensive experimental methods (X-ray crystallography, NMR) with fast, scalable deep learning inference.

**Prediction Tasks:**
- **Q8**: 8-class prediction 
- **Q3**: 3-class prediction

**10 Model Architectures**
1. **CNN**: Local pattern detection via 1D convolutions
2. **Transformer**: Long-range dependencies via self-attention
3. **CNN-Transformer**: Hybrid local + global modeling
4. **BiLSTM**: Bidirectional sequential modeling
5. **BiLSTM-CNN**: Combined sequential + local features

**Embedding Strategies:**
- **ESM2**: Pre-trained protein language model (640-dim per residue)
- **Learnable**: Train embeddings from scratch


## Notebooks

| # | Architecture | With ESM2 | Without ESM2 |
|---|--------------|-----------|--------------|
| 1 | **CNN** | `1_cnn_esm2.ipynb` | `1_cnn_no_embeddings.ipynb` |
| 2 | **Transformer** | `2_transformer_esm2.ipynb` | `2_transformer_no_embeddings.ipynb` |
| 3 | **CNN-Transformer** | `3_cnn_transformer_esm2.ipynb` | `3_cnn_transformer_no_embeddings.ipynb` |
| 4 | **BiLSTM** | `4_bilstm_esm2.ipynb` | `4_bilstm_no_embeddings.ipynb` |
| 5 | **BiLSTM-CNN** | `5_bilstm_cnn_esm2.ipynb` | `5_bilstm_cnn_no_embeddings.ipynb` |

## Dataset

- **12,857 protein chains** from PDB (PISCES culled, 2018-06-06)
- **Labels**: DSSP-assigned Q8 and Q3 secondary structures
- **Files**: `data/2018-06-06-pdb-intersect-pisces.csv`, `data/2018-06-06-ss.cleaned.csv`

### Data
- `data/2018-06-06-pdb-intersect-pisces.csv`: Protein sequence data
- `data/2018-06-06-ss.cleaned.csv`: Secondary structure labels (8-class and 3-class)

---

## Dataset

The project uses curated protein datasets with experimentally determined structures:
- **Sequences**: 12,857 protein chains from the PDB (Protein Data Bank)
- **Source**: PISCES culled PDB dataset (2018-06-06)
- **Labels**: DSSP-assigned secondary structure for both Q8 and Q3 classifications
- **Format**: CSV files with columns: `pdb_id`, `seq` (amino acid sequence), `sst8` (Q8 labels), `sst3` (Q3 labels)

---

## Quick Start

```bash
# Setup
git clone https://github.com/apaditya7/SC4001_Assignment2.git
cd SC4001_Assignment2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/
```

## Model Outputs

Each notebook produces:
- Best model checkpoint (selected by validation Q8 accuracy)
- Training/validation metrics (loss, accuracy per epoch)

## Evaluation Metrics

- Q8/Q3 accuracy
- SOV (Segment Overlap) for Q3

---

## Quick Start

1) Create and activate an environment, then install dependencies:
