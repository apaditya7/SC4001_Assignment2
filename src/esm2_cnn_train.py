import os
import json
import math
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_esm(model_name: str, device: torch.device):
    # Uses torch.hub to fetch ESM2; requires network at first download
    esm_model, alphabet = torch.hub.load(
        "facebookresearch/esm:main", model_name)
    esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter


def load_and_merge_csv(
    sequences_csv: str,
    labels_csv: str,
    id_col: str = 'pdb_id',
) -> pd.DataFrame:
    seq_df = pd.read_csv(sequences_csv)
    lab_df = pd.read_csv(labels_csv)
    if id_col not in seq_df.columns or id_col not in lab_df.columns:
        raise ValueError(f"id_col='{id_col}' must exist in both CSVs")
    df = pd.merge(seq_df, lab_df, on=id_col, how='inner')
    return df


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'seq' not in df.columns or 'sst8' not in df.columns or 'sst3' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'seq', 'sst8', 'sst3' columns")
    df['seq'] = df['seq'].astype(str).str.replace('*', 'X')
    df = df.dropna(subset=['seq', 'sst8', 'sst3']).copy()
    # keep rows where sequence and labels have same length
    mask = (df['seq'].str.len() == df['sst8'].str.len()) & (
        df['seq'].str.len() == df['sst3'].str.len())
    df = df[mask].reset_index(drop=True)
    return df


def compute_esm_embeddings(
    sequences: list,
    esm_model,
    batch_converter,
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[torch.Tensor, int]:
    data = list(zip(range(len(sequences)), [
                s.replace('*', 'X') for s in sequences]))
    all_embeddings = []
    for i in tqdm(range(0, len(data), batch_size), desc="Generating ESM2 embeddings"):
        batch_data = data[i:i+batch_size]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[
                                esm_model.num_layers], return_contacts=False)
        emb = results['representations'][esm_model.num_layers][:,
                                                               1:-1, :]  # strip BOS/EOS
        all_embeddings.extend([e.cpu() for e in emb])
    padded_embeddings = pad_sequence(
        all_embeddings, batch_first=True, padding_value=0.0)
    embedding_dim = padded_embeddings.shape[-1]
    return padded_embeddings, embedding_dim

# ---------------------------------------------------------------------------
# Memory-safe streaming: save per-sequence embeddings to disk
# ---------------------------------------------------------------------------


def save_esm_embeddings_to_dir(
    sequences: List[str],
    esm_model,
    batch_converter,
    device: torch.device,
    output_dir: str,
    batch_size: int = 8,
    dtype: str = 'float16',
) -> str:
    """Streams ESM2 embeddings to disk to avoid holding full tensor in RAM.

    Creates one .pt file per sequence and an index.csv with columns:
      idx,path,length,dim

    Returns path to index.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_rows = []
    torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
    data = list(zip(range(len(sequences)), [
                s.replace('*', 'X') for s in sequences]))
    for i in tqdm(range(0, len(data), batch_size), desc="Saving ESM2 embeddings"):
        batch = data[i:i+batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[
                                esm_model.num_layers], return_contacts=False)
        reps = results['representations'][esm_model.num_layers][:, 1:-1, :]
        for j, emb in enumerate(reps):
            global_idx = i + j
            emb_cpu = emb.detach().to('cpu').to(dtype=torch_dtype)
            L, D = emb_cpu.shape
            out_path = os.path.join(output_dir, f"{global_idx:07d}.pt")
            torch.save(emb_cpu, out_path)
            index_rows.append(
                {'idx': global_idx, 'path': out_path, 'length': int(L), 'dim': int(D)})
    import pandas as _pd  # local import to avoid shadowing user imports
    index_df = _pd.DataFrame(index_rows).sort_values(
        'idx').reset_index(drop=True)
    index_path = os.path.join(output_dir, 'index.csv')
    index_df.to_csv(index_path, index=False)
    return index_path


class PrecomputedEmbeddingDataset(Dataset):
    """Lazy-loads per-sequence embeddings + builds label tensors on demand."""

    def __init__(self, index_df: pd.DataFrame, df: pd.DataFrame, ss8_vocab: Dict[str, int], ss3_vocab: Dict[str, int]):
        self.index_df = index_df
        self.df = df
        self.ss8_vocab = ss8_vocab
        self.ss3_vocab = ss3_vocab

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, i):
        row = self.index_df.iloc[i]
        emb: torch.Tensor = torch.load(
            row['path'], map_location='cpu')  # [L, D]
        L = emb.shape[0]
        s8 = str(self.df.iloc[i]['sst8'])
        s3 = str(self.df.iloc[i]['sst3'])
        s8_ids = [self.ss8_vocab.get(c, -1) for c in s8][:L]
        s3_ids = [self.ss3_vocab.get(c, -1) for c in s3][:L]
        return emb, torch.tensor(s8_ids, dtype=torch.long), torch.tensor(s3_ids, dtype=torch.long)


def collate_embeddings(batch):
    embs, s8s, s3s = zip(*batch)
    padded_embs = pad_sequence(embs, batch_first=True, padding_value=0.0)
    padded_s8 = pad_sequence(s8s, batch_first=True, padding_value=-1)
    padded_s3 = pad_sequence(s3s, batch_first=True, padding_value=-1)
    return padded_embs, padded_s8, padded_s3


def encode_labels(ss_labels, vocab, max_len):
    encoded = []
    for ss in ss_labels:
        ids = [vocab.get(c, -1) for c in str(ss)]
        if len(ids) < max_len:
            ids.extend([-1] * (max_len - len(ids)))
        else:
            ids = ids[:max_len]
        encoded.append(torch.tensor(ids, dtype=torch.long))
    return pad_sequence(encoded, batch_first=True, padding_value=-1)


class ProteinCNNOnEmb(nn.Module):
    def __init__(self, input_dim=640, num_filters=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=num_filters, out_channels=num_filters, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(
            in_channels=num_filters, out_channels=num_filters, kernel_size=7, padding=3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.q8_head = nn.Linear(num_filters, 8)
        self.q3_head = nn.Linear(num_filters, 3)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.dropout2(self.relu2(self.conv2(x)))
        x = self.dropout3(self.relu3(self.conv3(x)))
        x = x.permute(0, 2, 1)  # [B, L, C]
        return self.q8_head(x), self.q3_head(x)


@torch.no_grad()
def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels >= 0
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion_q8 = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
    criterion_q3 = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    clip_norm = 1.0

    tr_loss = tr_acc8 = tr_acc3 = 0.0
    for emb, s8, s3 in tqdm(loader, leave=False):
        emb, s8, s3 = emb.to(device), s8.to(device), s3.to(device)
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast(device_type='cuda') if torch.cuda.is_available(
        ) else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            q8, q3 = model(emb)
            l8 = criterion_q8(q8.view(-1, 8), s8.view(-1))
            l3 = criterion_q3(q3.view(-1, 3), s3.view(-1))
            loss = l8 + 0.5 * l3
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item()
        tr_acc8 += compute_accuracy(q8, s8)
        tr_acc3 += compute_accuracy(q3, s3)
    n = max(1, len(loader))
    return tr_loss / n, tr_acc8 / n, tr_acc3 / n


def evaluate(model, loader, device):
    model.eval()
    criterion_q8 = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
    criterion_q3 = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
    va_loss = va_acc8 = va_acc3 = 0.0
    with torch.no_grad():
        for emb, s8, s3 in loader:
            emb, s8, s3 = emb.to(device), s8.to(device), s3.to(device)
            with (torch.amp.autocast(device_type='cuda') if torch.cuda.is_available() else torch.cuda.amp.autocast(enabled=False)):
                q8, q3 = model(emb)
                l8 = criterion_q8(q8.view(-1, 8), s8.view(-1))
                l3 = criterion_q3(q3.view(-1, 3), s3.view(-1))
                loss = l8 + 0.5 * l3
            va_loss += loss.item()
            va_acc8 += compute_accuracy(q8, s8)
            va_acc3 += compute_accuracy(q3, s3)
    n = max(1, len(loader))
    return va_loss / n, va_acc8 / n, va_acc3 / n


def train_esm2_cnn(
    # data/config
    config_path: Optional[str] = None,
    sequences_csv: Optional[str] = None,
    labels_csv: Optional[str] = None,
    id_column: str = 'pdb_id',
    output_dir: str = 'checkpoints',
    # esm
    esm_model_name: str = 'esm2_t30_150M_UR50D',
    esm_batch_size: int = 8,
    # model
    num_filters: int = 128,
    dropout: float = 0.1,
    # training
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    patience: int = 7,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[str] = None,
    dry_run: bool = False,
    # memory / precompute options
    embeddings_dir: Optional[str] = None,
    precompute: bool = False,
    precompute_only: bool = False,
    save_dtype: str = 'float16',
) -> Dict[str, Any]:
    """Train CNN on ESM2 embeddings."""
    # Config merge
    if config_path and yaml is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        seed = cfg.get('seed', seed)
        data_cfg = cfg.get('data', {})
        sequences_csv = sequences_csv or data_cfg.get('sequences_csv')
        labels_csv = labels_csv or data_cfg.get('labels_csv')
        id_column = data_cfg.get('id_column', id_column)
        out_cfg = cfg.get('output', {})
        output_dir = out_cfg.get('checkpoints_dir', output_dir)
        train_cfg = cfg.get('training', {})
        epochs = train_cfg.get('epochs', epochs)
        batch_size = train_cfg.get('batch_size', batch_size)
        lr = train_cfg.get('lr', lr)
        num_workers = train_cfg.get('num_workers', num_workers)
        model_cfg = cfg.get('model', {})
        dropout = model_cfg.get('dropout', dropout)
        num_filters = model_cfg.get('hidden_dim', num_filters)

    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_t = torch.device(device)

    if sequences_csv is None or labels_csv is None:
        # best-effort defaults like in the notebook
        repo_root = os.path.dirname(os.path.dirname(__file__))
        sequences_csv = sequences_csv or os.path.join(
            repo_root, 'data', '2018-06-06-pdb-intersect-pisces.csv')
        labels_csv = labels_csv or os.path.join(
            repo_root, 'data', '2018-06-06-ss.cleaned.csv')

    df = load_and_merge_csv(sequences_csv, labels_csv, id_col=id_column)
    df = sanitize_df(df)

    if dry_run:
        return {
            'status': 'dry-run',
            'num_rows': len(df),
            'sequences_csv': sequences_csv,
            'labels_csv': labels_csv,
            'output_dir': output_dir,
        }

    # Common vocab
    ss8_vocab = {'H': 0, 'G': 1, 'I': 2,
                 'E': 3, 'B': 4, 'T': 5, 'S': 6, 'C': 7}
    ss3_vocab = {'H': 0, 'E': 1, 'C': 2}

    # Branch 1: existing precomputed directory supplied
    if embeddings_dir and os.path.exists(os.path.join(embeddings_dir, 'index.csv')):
        index_df = pd.read_csv(os.path.join(embeddings_dir, 'index.csv'))
        embedding_dim = int(index_df['dim'].max())
        idx_all = list(range(len(index_df)))
        train_idx, temp_idx = train_test_split(
            idx_all, test_size=0.2, random_state=seed)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed)
        train_ds = PrecomputedEmbeddingDataset(index_df.loc[train_idx].reset_index(
            drop=True), df.loc[train_idx].reset_index(drop=True), ss8_vocab, ss3_vocab)
        val_ds = PrecomputedEmbeddingDataset(index_df.loc[val_idx].reset_index(
            drop=True),   df.loc[val_idx].reset_index(drop=True),   ss8_vocab, ss3_vocab)
        test_ds = PrecomputedEmbeddingDataset(index_df.loc[test_idx].reset_index(
            drop=True),  df.loc[test_idx].reset_index(drop=True),  ss8_vocab, ss3_vocab)
        loader_opts = dict(num_workers=num_workers,
                           pin_memory=True, collate_fn=collate_embeddings)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **loader_opts)
        val_loader = DataLoader(
            val_ds,   batch_size=batch_size, shuffle=False, **loader_opts)
        test_loader = DataLoader(
            test_ds,  batch_size=batch_size, shuffle=False, **loader_opts)
        model = ProteinCNNOnEmb(input_dim=embedding_dim,
                                num_filters=num_filters, dropout=dropout)
    else:
        # Need to generate (stream or in-memory)
        esm_model, batch_converter = load_esm(esm_model_name, device_t)
        if precompute or embeddings_dir or precompute_only:
            out_dir = embeddings_dir or os.path.join(
                output_dir, 'esm2_embeddings')
            index_path = save_esm_embeddings_to_dir(df['seq'].tolist(
            ), esm_model, batch_converter, device_t, out_dir, batch_size=esm_batch_size, dtype=save_dtype)
            if precompute_only:
                return {
                    'status': 'precomputed',
                    'embeddings_dir': out_dir,
                    'index_path': index_path,
                    'output_dir': output_dir,
                }
            index_df = pd.read_csv(index_path)
            embedding_dim = int(index_df['dim'].max())
            idx_all = list(range(len(index_df)))
            train_idx, temp_idx = train_test_split(
                idx_all, test_size=0.2, random_state=seed)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=seed)
            train_ds = PrecomputedEmbeddingDataset(index_df.loc[train_idx].reset_index(
                drop=True), df.loc[train_idx].reset_index(drop=True), ss8_vocab, ss3_vocab)
            val_ds = PrecomputedEmbeddingDataset(index_df.loc[val_idx].reset_index(
                drop=True),   df.loc[val_idx].reset_index(drop=True),   ss8_vocab, ss3_vocab)
            test_ds = PrecomputedEmbeddingDataset(index_df.loc[test_idx].reset_index(
                drop=True),  df.loc[test_idx].reset_index(drop=True),  ss8_vocab, ss3_vocab)
            loader_opts = dict(num_workers=num_workers,
                               pin_memory=True, collate_fn=collate_embeddings)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, **loader_opts)
            val_loader = DataLoader(
                val_ds,   batch_size=batch_size, shuffle=False, **loader_opts)
            test_loader = DataLoader(
                test_ds,  batch_size=batch_size, shuffle=False, **loader_opts)
            model = ProteinCNNOnEmb(
                input_dim=embedding_dim, num_filters=num_filters, dropout=dropout)
        else:
            # In-memory (fallback); may OOM for very large sets
            padded_embeddings, embedding_dim = compute_esm_embeddings(df['seq'].tolist(
            ), esm_model, batch_converter, device_t, batch_size=esm_batch_size)
            seq_pad_len = padded_embeddings.shape[1]
            ss8_labels = encode_labels(df['sst8'], ss8_vocab, seq_pad_len)
            ss3_labels = encode_labels(df['sst3'], ss3_vocab, seq_pad_len)
            idx_all = list(range(len(padded_embeddings)))
            train_idx, temp_idx = train_test_split(
                idx_all, test_size=0.2, random_state=seed)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=seed)
            train_ds = TensorDataset(
                padded_embeddings[train_idx], ss8_labels[train_idx], ss3_labels[train_idx])
            val_ds = TensorDataset(
                padded_embeddings[val_idx],   ss8_labels[val_idx],   ss3_labels[val_idx])
            test_ds = TensorDataset(
                padded_embeddings[test_idx],  ss8_labels[test_idx],  ss3_labels[test_idx])
            loader_opts = dict(num_workers=num_workers, pin_memory=True)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, **loader_opts)
            val_loader = DataLoader(
                val_ds,   batch_size=batch_size, shuffle=False, **loader_opts)
            test_loader = DataLoader(
                test_ds,  batch_size=batch_size, shuffle=False, **loader_opts)
            model = ProteinCNNOnEmb(
                input_dim=embedding_dim, num_filters=num_filters, dropout=dropout)
    if torch.cuda.device_count() > 1 and device_t.type == 'cuda':
        model = nn.DataParallel(model)
    model.to(device_t)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3)

    best_val = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(output_dir, 'best_esm2_cnn.pt')

    history = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc8, tr_acc3 = train_one_epoch(
            model, train_loader, optimizer, device_t)
        va_loss, va_acc8, va_acc3 = evaluate(model, val_loader, device_t)
        scheduler.step(va_loss)
        tqdm.write(
            f"Epoch {epoch:03d}: TrainLoss={tr_loss:.4f} ValLoss={va_loss:.4f} | "
            f"TrainQ8={tr_acc8:.4f} ValQ8={va_acc8:.4f} | TrainQ3={tr_acc3:.4f} ValQ3={va_acc3:.4f}"
        )
        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': va_loss,
            'train_acc_q8': tr_acc8,
            'val_acc_q8': va_acc8,
            'train_acc_q3': tr_acc3,
            'val_acc_q3': va_acc3,
        })
        metric = va_acc8
        if metric > best_val:
            best_val = metric
            epochs_no_improve = 0
            state = model.module.state_dict() if isinstance(
                model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write('Early stopping triggered.')
                break

    # Test
    best_model = ProteinCNNOnEmb(
        input_dim=embedding_dim, num_filters=num_filters, dropout=dropout)
    best_state = torch.load(best_path, map_location=device_t)
    best_model.load_state_dict(best_state)
    if torch.cuda.device_count() > 1 and device_t.type == 'cuda':
        best_model = nn.DataParallel(best_model)
    best_model.to(device_t)
    te_loss, te_acc8, te_acc3 = evaluate(best_model, test_loader, device_t)

    metrics = {
        'best_val_acc_q8': best_val,
        'test_loss': te_loss,
        'test_acc_q8': te_acc8,
        'test_acc_q3': te_acc3,
    }
    metrics_path = os.path.join(output_dir, 'metrics_esm2_cnn.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {
        'status': 'ok',
        'history': history,
        'metrics': metrics,
        'best_model_path': best_path,
        'metrics_path': metrics_path,
        'output_dir': output_dir,
    }


def _resolve_default_config_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(repo_root, 'config.yaml')


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train CNN on ESM2 embeddings')
    parser.add_argument('--config', type=str,
                        default=_resolve_default_config_path())
    parser.add_argument('--sequences_csv', type=str, default=None)
    parser.add_argument('--labels_csv', type=str, default=None)
    parser.add_argument('--id_column', type=str, default='pdb_id')
    parser.add_argument('--output_dir', type=str, default='checkpoints')

    parser.add_argument('--esm_model_name', type=str,
                        default='esm2_t30_150M_UR50D')
    parser.add_argument('--esm_batch_size', type=int, default=8)

    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                        help='Directory containing or to store streamed embeddings')
    parser.add_argument('--precompute', action='store_true',
                        help='Stream ESM embeddings to disk then train from them')
    parser.add_argument('--precompute_only', action='store_true',
                        help='Only generate embeddings, do not train')
    parser.add_argument('--save_dtype', type=str, default='float16',
                        choices=['float16', 'float32'], help='On-disk embedding dtype')

    args = parser.parse_args()

    result = train_esm2_cnn(
        config_path=args.config,
        sequences_csv=args.sequences_csv,
        labels_csv=args.labels_csv,
        id_column=args.id_column,
        output_dir=args.output_dir,
        esm_model_name=args.esm_model_name,
        esm_batch_size=args.esm_batch_size,
        num_filters=args.num_filters,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        dry_run=args.dry_run,
        embeddings_dir=args.embeddings_dir,
        precompute=args.precompute,
        precompute_only=args.precompute_only,
        save_dtype=args.save_dtype,
    )

    print(json.dumps(
        {k: v for k, v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
