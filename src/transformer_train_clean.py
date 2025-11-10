import os
import math
import json
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Data Loading / Preprocessing
# -----------------------------
def load_sequence_label_dataframe(
    data_csv: Optional[str] = None,
    sequences_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads a dataframe with at least columns: 'seq', 'sst8', 'sst3'.
    Priority order:
      1) data_csv (single file containing seq+labels)
      2) labels_csv (if it contains both seq and labels)
      3) sequences_csv (fallback; must contain labels too)
    """
    candidate_paths = [p for p in [data_csv, labels_csv, sequences_csv] if p]
    if not candidate_paths:
        # reasonable default from repo layout
        default = os.path.join(os.path.dirname(os.path.dirname(
            __file__)), "data", "2018-06-06-ss.cleaned.csv")
        candidate_paths = [default]

    last_err = None
    for p in candidate_paths:
        try:
            df = pd.read_csv(p)
            # Basic sanity checks
            if 'seq' not in df.columns:
                raise ValueError(f"CSV {p} missing 'seq' column")
            if not (('sst8' in df.columns) and ('sst3' in df.columns)):
                raise ValueError(f"CSV {p} missing 'sst8'/'sst3' columns")
            # Preprocess
            if 'len' not in df.columns:
                df['len'] = df['seq'].astype(str).str.len()
            df['seq'] = df['seq'].astype(str).str.replace('*', 'X')
            if 'has_nonstd_aa' in df.columns:
                df = df[df['has_nonstd_aa'] == False].reset_index(drop=True)
            return df
        except Exception as e:  # keep trying others
            last_err = e
            continue
    # If we got here, everything failed
    raise RuntimeError(
        f"Failed to load a usable CSV from {candidate_paths}: {last_err}")


def build_vocabs(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Builds vocab for amino acids (with <pad>=0) and label vocabs for q8 and q3."""
    ss8_vocab = {'H': 0, 'G': 1, 'I': 2,
                 'E': 3, 'B': 4, 'T': 5, 'S': 6, 'C': 7}
    ss3_vocab = {'H': 0, 'E': 1, 'C': 2}

    all_chars = set(''.join(df['seq'].astype(str).tolist()))
    seq_vocab = {ch: i + 1 for i, ch in enumerate(sorted(list(all_chars)))}
    seq_vocab['<pad>'] = 0
    return seq_vocab, ss8_vocab, ss3_vocab


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, sst8_labels, sst3_labels, seq_vocab, ss8_vocab, ss3_vocab):
        self.sequences = sequences
        self.sst8_labels = sst8_labels
        self.sst3_labels = sst3_labels
        self.seq_vocab = seq_vocab
        self.ss8_vocab = ss8_vocab
        self.ss3_vocab = ss3_vocab

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        ss8 = str(self.sst8_labels[idx])
        ss3 = str(self.sst3_labels[idx])

        seq_tokens = [self.seq_vocab.get(c, 0) for c in seq]
        ss8_tokens = [self.ss8_vocab.get(c, -1)
                      for c in ss8][: len(seq_tokens)]
        ss3_tokens = [self.ss3_vocab.get(c, -1)
                      for c in ss3][: len(seq_tokens)]

        return (
            torch.tensor(seq_tokens, dtype=torch.long),
            torch.tensor(ss8_tokens, dtype=torch.long),
            torch.tensor(ss3_tokens, dtype=torch.long),
        )


def make_collate_fn(seq_vocab: Dict[str, int]):
    pad_id = seq_vocab['<pad>']

    def collate_fn(batch):
        seqs, ss8s, ss3s = zip(*batch)
        padded_seqs = pad_sequence(
            seqs, batch_first=True, padding_value=pad_id)
        padded_ss8s = pad_sequence(ss8s, batch_first=True, padding_value=-1)
        padded_ss3s = pad_sequence(ss3s, batch_first=True, padding_value=-1)
        return padded_seqs, padded_ss8s, padded_ss3s

    return collate_fn


# -----------------------------
# Model
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ProteinTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(
            vocab_size, input_dim, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.q8_head = nn.Linear(input_dim, 8)
        self.q3_head = nn.Linear(input_dim, 3)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.input_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        q8_logits = self.q8_head(x)
        q3_logits = self.q3_head(x)
        return q8_logits, q3_logits


# -----------------------------
# Training / Evaluation
# -----------------------------
@torch.no_grad()
def compute_accuracy(pred_logits, labels) -> float:
    preds = pred_logits.argmax(-1)
    mask = labels != -1
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device, seq_pad_token):
    model.train()
    criterion_q8 = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_q3 = nn.CrossEntropyLoss(ignore_index=-1)

    epoch_loss = 0.0
    epoch_acc_q8 = 0.0
    epoch_acc_q3 = 0.0

    for seqs, ss8, ss3 in tqdm(loader, leave=False):
        seqs = seqs.to(device)
        ss8 = ss8.to(device)
        ss3 = ss3.to(device)
        mask = (seqs == seq_pad_token)

        q8_logits, q3_logits = model(seqs, mask)
        loss_q8 = criterion_q8(q8_logits.view(-1, 8), ss8.view(-1))
        loss_q3 = criterion_q3(q3_logits.view(-1, 3), ss3.view(-1))
        loss = loss_q8 + 0.5 * loss_q3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc_q8 += compute_accuracy(q8_logits, ss8)
        epoch_acc_q3 += compute_accuracy(q3_logits, ss3)

    n = max(1, len(loader))
    return epoch_loss / n, epoch_acc_q8 / n, epoch_acc_q3 / n


def evaluate(model, loader, device, seq_pad_token):
    model.eval()
    criterion_q8 = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_q3 = nn.CrossEntropyLoss(ignore_index=-1)

    epoch_loss = 0.0
    epoch_acc_q8 = 0.0
    epoch_acc_q3 = 0.0

    with torch.no_grad():
        for seqs, ss8, ss3 in loader:
            seqs = seqs.to(device)
            ss8 = ss8.to(device)
            ss3 = ss3.to(device)
            mask = (seqs == seq_pad_token)

            q8_logits, q3_logits = model(seqs, mask)
            loss_q8 = criterion_q8(q8_logits.view(-1, 8), ss8.view(-1))
            loss_q3 = criterion_q3(q3_logits.view(-1, 3), ss3.view(-1))
            loss = loss_q8 + 0.5 * loss_q3

            epoch_loss += loss.item()
            epoch_acc_q8 += compute_accuracy(q8_logits, ss8)
            epoch_acc_q3 += compute_accuracy(q3_logits, ss3)

    n = max(1, len(loader))
    return epoch_loss / n, epoch_acc_q8 / n, epoch_acc_q3 / n


# -----------------------------
# Orchestrator
# -----------------------------
def train_transformer_clean(
    # data/config
    config_path: Optional[str] = None,
    data_csv: Optional[str] = None,
    sequences_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    output_dir: str = "checkpoints",
    # model params
    embedding_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 8,
    ff_dim: int = 512,
    dropout: float = 0.1,
    # training
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate the Transformer from the notebook as a function.

    Returns a dict with summary metrics and artifact paths.
    """
    # Parse config if provided (allows overriding defaults)
    if config_path and yaml is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        seed = cfg.get('seed', seed)
        data_cfg = cfg.get('data', {})
        sequences_csv = data_cfg.get('sequences_csv', sequences_csv)
        labels_csv = data_cfg.get('labels_csv', labels_csv)
        # prefer a single CSV that contains labels
        data_csv = data_csv or data_cfg.get(
            'labels_csv') or data_cfg.get('sequences_csv')

        train_cfg = cfg.get('training', {})
        epochs = train_cfg.get('epochs', epochs)
        batch_size = train_cfg.get('batch_size', batch_size)
        lr = train_cfg.get('lr', lr)
        num_workers = train_cfg.get('num_workers', num_workers)

        model_cfg = cfg.get('model', {})
        embedding_dim = model_cfg.get('embedding_dim', embedding_dim)
        dropout = model_cfg.get('dropout', dropout)
        # reuse 'layers' as number of transformer layers if present
        num_layers = model_cfg.get('layers', num_layers)

        out_cfg = cfg.get('output', {})
        output_dir = out_cfg.get('checkpoints_dir', output_dir)

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_t = torch.device(device)

    # Load data
    df = load_sequence_label_dataframe(
        data_csv=data_csv, sequences_csv=sequences_csv, labels_csv=labels_csv)

    # Build vocabs
    seq_vocab, ss8_vocab, ss3_vocab = build_vocabs(df)
    collate_fn = make_collate_fn(seq_vocab)
    pad_id = seq_vocab['<pad>']

    # Splits
    train_indices, temp_indices = train_test_split(
        range(len(df)), test_size=0.2, random_state=seed)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=seed)

    def make_ds(idxs): return ProteinSequenceDataset(
        df.iloc[idxs]['seq'].tolist(),
        df.iloc[idxs]['sst8'].tolist(),
        df.iloc[idxs]['sst3'].tolist(),
        seq_vocab,
        ss8_vocab,
        ss3_vocab,
    )

    train_loader = DataLoader(make_ds(train_indices), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(make_ds(val_indices), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(make_ds(test_indices), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Model
    vocab_size = len(seq_vocab)
    model = ProteinTransformer(
        vocab_size=vocab_size,
        input_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pad_id=pad_id,
    )

    if torch.cuda.device_count() > 1 and device_t.type == 'cuda':
        model = nn.DataParallel(model)

    model.to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc_q8 = -float('inf')
    best_model_path = os.path.join(output_dir, 'best_transformer_clean.pt')

    if dry_run:
        # quick pass-through to ensure pipeline builds
        return {
            'status': 'dry-run',
            'vocab_size': vocab_size,
            'num_train': len(train_indices),
            'num_val': len(val_indices),
            'num_test': len(test_indices),
            'output_dir': output_dir,
            'best_model_path': best_model_path,
        }

    # Train
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_q8, train_acc_q3 = train_one_epoch(
            model, train_loader, optimizer, device_t, pad_id)
        val_loss, val_acc_q8, val_acc_q3 = evaluate(
            model, val_loader, device_t, pad_id)

        tqdm.write(
            f"Epoch {epoch:03d}: TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} | "
            f"TrainQ8={train_acc_q8:.4f} ValQ8={val_acc_q8:.4f} | TrainQ3={train_acc_q3:.4f} ValQ3={val_acc_q3:.4f}"
        )

        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc_q8': train_acc_q8,
                'val_acc_q8': val_acc_q8,
                'train_acc_q3': train_acc_q3,
                'val_acc_q3': val_acc_q3,
            }
        )

        if val_acc_q8 > best_val_acc_q8:
            state = model.module.state_dict() if isinstance(
                model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_model_path)
            best_val_acc_q8 = val_acc_q8

    # Test with best model
    best_model = ProteinTransformer(
        vocab_size=vocab_size,
        input_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pad_id=pad_id,
    )
    best_state = torch.load(best_model_path, map_location=device_t)
    best_model.load_state_dict(best_state)

    if torch.cuda.device_count() > 1 and device_t.type == 'cuda':
        best_model = nn.DataParallel(best_model)
    best_model.to(device_t)

    test_loss, test_acc_q8, test_acc_q3 = evaluate(
        best_model, test_loader, device_t, pad_id)

    metrics = {
        'best_val_acc_q8': best_val_acc_q8,
        'test_loss': test_loss,
        'test_acc_q8': test_acc_q8,
        'test_acc_q3': test_acc_q3,
    }

    metrics_path = os.path.join(output_dir, 'metrics_transformer_clean.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {
        'status': 'ok',
        'history': history,
        'metrics': metrics,
        'best_model_path': best_model_path,
        'metrics_path': metrics_path,
        'output_dir': output_dir,
    }


def _resolve_default_config_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(repo_root, 'config.yaml')


def main():  # CLI
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Transformer (clean) from notebook as a script')
    parser.add_argument(
        '--config', type=str, default=_resolve_default_config_path(), help='Path to YAML config')
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Single CSV with seq and labels')
    parser.add_argument('--sequences_csv', type=str, default=None)
    parser.add_argument('--labels_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints')

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    result = train_transformer_clean(
        config_path=args.config,
        data_csv=args.data_csv,
        sequences_csv=args.sequences_csv,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        dry_run=args.dry_run,
    )

    # print compact summary for logs
    print(json.dumps(
        {k: v for k, v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
