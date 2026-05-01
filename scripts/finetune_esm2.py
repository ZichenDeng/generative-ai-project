#!/usr/bin/env python3
"""
Fine-tune ESM-2 on FLAb antibody fitness tasks

run:
    python scripts/finetune_esm2.py --tasks koenig_binding_g6 koenig_expression_g6 warszawski_binding_d44

- freeze all ESM-2 layers except last N transformer blocks + LM head
- add lightweight regression head (LayerNorm -> Linear -> GELU -> Linear)
- concatenate heavy + light chain mean-pooled embeddings as input to head
- train w/ MSE loss, eval w/ Spearman/Pearson each epoch
- use same 10-fold CV splits as the frozen-embedding baselines
- save best checkpoint per fold (by val Spearman) to results/checkpoints/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel, get_linear_schedule_with_warmup


TASKS = {
    "koenig_binding_g6": "data/processed/koenig2017mutational_kd_g6_folds.csv",
    "koenig_expression_g6": "data/processed/koenig2017mutational_er_g6_folds.csv",
    "warszawski_binding_d44": "data/processed/warszawski2019_d44_Kd_folds.csv",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# data
def load_task(path, label_column, fold_column, max_rows=None):
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["heavy", label_column, fold_column]).copy()
    frame["light"] = frame["light"].fillna("")
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame.dropna(subset=[label_column]).copy()
    frame[fold_column] = frame[fold_column].astype(int)
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    return frame


class AntibodyDataset(Dataset):
    """
    return (heavy_seq, light_seq, fitness) tuples
    """
    def __init__(self, frame, label_column):
        self.heavy = frame["heavy"].astype(str).tolist()
        self.light = frame["light"].astype(str).tolist()
        self.labels = frame[label_column].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.heavy[idx], self.light[idx], self.labels[idx]


def collate_fn(batch, tokenizer, max_length=512):
    heavy_seqs, light_seqs, labels = zip(*batch)
    heavy_enc = tokenizer(
        list(heavy_seqs),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    light_enc = tokenizer(
        list(light_seqs),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return heavy_enc, light_enc, labels_tensor


# model
class ESM2RegressionHead(nn.Module):
    """
    small regression head on top of concatenated heavy+light embeddings
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        input_dim = hidden_size * 2  # heavy || light
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FineTunedESM2(nn.Module):
    def __init__(self, esm2: EsmModel, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.esm2 = esm2
        self.head = ESM2RegressionHead(hidden_size, dropout)

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask, special_tokens_mask):
        valid = attention_mask.bool() & ~special_tokens_mask.bool()
        valid = valid.unsqueeze(-1).float()
        summed = (last_hidden_state * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1)
        return summed / counts

    def encode(self, enc):
        out = self.esm2(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        return self.mean_pool(
            out.last_hidden_state,
            enc["attention_mask"],
            enc["special_tokens_mask"],
        )

    def forward(self, heavy_enc, light_enc):
        h = self.encode(heavy_enc)
        l = self.encode(light_enc)
        # zero-out light embedding for empty light chains (all padding)
        # identified by attention_mask summing to 0 after removing [CLS]/[EOS]
        light_valid = (light_enc["attention_mask"] & ~light_enc["special_tokens_mask"]).any(dim=1)
        l = l * light_valid.unsqueeze(-1).float()
        return self.head(torch.cat([h, l], dim=-1))


def freeze_esm2_layers(model: EsmModel, n_unfreeze: int):
    """
    freeze all ESM-2 params except the last n_unfreeze encoder layers
    """
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last n_unfreeze transformer blocks
    n_layers = len(model.encoder.layer)
    for i in range(n_layers - n_unfreeze, n_layers):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = True

    # always unfreeze final LayerNorm
    if hasattr(model, "pooler") and model.pooler is not None:
        for param in model.pooler.parameters():
            param.requires_grad = True
    if hasattr(model.encoder, "emb_layer_norm_after"):
        for param in model.encoder.emb_layer_norm_after.parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  ESM-2 trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")


# training/eval
def move_enc(enc, device):
    return {k: v.to(device) for k, v in enc.items()}


def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    # use mse loss
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    for heavy_enc, light_enc, labels in loader:
        heavy_enc = move_enc(heavy_enc, device)
        light_enc = move_enc(light_enc, device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                preds = model(heavy_enc, light_enc)
                loss = loss_fn(preds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(heavy_enc, light_enc)
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()  # after optimizer.step()
        scheduler.step()  # after optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for heavy_enc, light_enc, labels in loader:
        heavy_enc = move_enc(heavy_enc, device)
        light_enc = move_enc(light_enc, device)
        preds = model(heavy_enc, light_enc).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    # compute stats
    spearman = spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else np.nan
    pearson = pearsonr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"spearman": spearman, "pearson": pearson, "rmse": rmse, "mae": mae, "r2": r2,
            "y_pred": y_pred, "y_true": y_true}


# cross-validation loop
def run_finetuning_cv(task_id, frame, args, tokenizer, device):
    label_col = args.label_column
    fold_col = args.fold_column
    folds = sorted(frame[fold_col].unique())
    checkpoint_dir = repo_root() / "results" / "checkpoints" / task_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import functools
    collate = functools.partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    rows = []

    for fold in folds:
        print(f"\n  --- Fold {fold} ---")
        train_frame = frame[frame[fold_col] != fold].copy()
        val_frame = frame[frame[fold_col] == fold].copy()

        train_ds = AntibodyDataset(train_frame, label_col)
        val_ds = AntibodyDataset(val_frame, label_col)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            collate_fn=collate, num_workers=0,
        )

        # build fresh model for each fold
        print(f"  Loading ESM-2 ({args.model_name})")
        esm2 = EsmModel.from_pretrained(args.model_name).to(device)
        freeze_esm2_layers(esm2, n_unfreeze=args.unfreeze_layers)
        hidden_size = esm2.config.hidden_size
        model = FineTunedESM2(esm2, hidden_size, dropout=args.dropout).to(device)

        # optimizer
        # separate LRs for backbone vs head
        head_params = list(model.head.parameters())
        backbone_params = [p for p in model.esm2.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ], weight_decay=args.weight_decay)

        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        use_amp = (device.type == "cuda") and args.use_amp
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_spearman = -np.inf
        best_ckpt_path = checkpoint_dir / f"fold{fold}_best.pt"

        # spearman as main metric
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
            val_metrics = evaluate(model, val_loader, device)
            sp = val_metrics["spearman"]
            print(
                f"  Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}"
                f"  val_spearman={sp:.4f}  val_pearson={val_metrics['pearson']:.4f}"
                f"  val_rmse={val_metrics['rmse']:.4f}"
            )
            if sp > best_spearman:
                best_spearman = sp
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "val_metrics": val_metrics},
                    best_ckpt_path,
                )

        # reload best checkpoint for final eval
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        final_metrics = evaluate(model, val_loader, device)
        print(
            f"  Best (epoch {ckpt['epoch']}): spearman={final_metrics['spearman']:.4f}"
            f"  pearson={final_metrics['pearson']:.4f}  rmse={final_metrics['rmse']:.4f}"
        )
        rows.append({
            "task": task_id,
            "model": f"esm2_finetuned_{args.unfreeze_layers}layers",
            "fold": fold,
            "n_train": len(train_frame),
            "n_test": len(val_frame),
            "best_epoch": ckpt["epoch"],
            **{k: v for k, v in final_metrics.items() if k not in ("y_pred", "y_true")},
        })

        del model, esm2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # aggregate mean
    mean_row = {
        "task": task_id,
        "model": f"esm2_finetuned_{args.unfreeze_layers}layers",
        "fold": "mean",
        "n_train": sum(r["n_train"] for r in rows),
        "n_test": sum(r["n_test"] for r in rows),
        "best_epoch": "—",
    }
    for metric in ("spearman", "pearson", "rmse", "mae", "r2"):
        mean_row[metric] = float(np.nanmean([r[metric] for r in rows]))
    rows.append(mean_row)
    return rows


# outputs
def write_outputs(rows, output_csv, output_md):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)

    summary = frame[frame["fold"].astype(str) == "mean"].copy()
    with open(output_md, "w") as f:
        f.write("# ESM-2 Fine-Tuning Results\n\n")
        f.write("Fine-tuned ESM-2 (last N transformer layers unfrozen) + regression head.\n\n")
        f.write("| Task | Model | Spearman | Pearson | RMSE | MAE | R2 |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for _, row in summary.sort_values(["task", "model"]).iterrows():
            f.write(
                f"| {row['task']} | {row['model']} | {row['spearman']:.4f}"
                f" | {row['pearson']:.4f} | {row['rmse']:.4f}"
                f" | {row['mae']:.4f} | {row['r2']:.4f} |\n"
            )
    print(f"\nWrote {output_csv}")
    print(f"Wrote {output_md}")


#args
def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Fine-tune ESM-2 on FLAb tasks.")
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D",
                        help="HF model name. Use esm2_t12_35M or esm2_t30_150M for bigger models.")
    parser.add_argument("--tasks", nargs="+",
                        default=["koenig_binding_g6", "koenig_expression_g6"],
                        choices=list(TASKS.keys()))
    parser.add_argument("--include-warszawski", action="store_true")
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument("--fold-column", default="cv_fold")
    # Fine-tuning
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="LR for unfrozen ESM-2 layers.")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                        help="LR for regression head.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-layers", type=int, default=2,
                        help="Number of last ESM-2 transformer blocks to unfreeze. "
                             "0 = head-only (linear probe). 2 is a good default.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length per chain (truncate longer sequences).")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Subsample for quick testing.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-csv",
                        default=str(root / "results" / "esm2-finetuned-results.csv"))
    parser.add_argument("--output-md",
                        default=str(root / "results" / "esm2-finetuned-results.md"))
    args = parser.parse_args()
    if args.include_warszawski and "warszawski_binding_d44" not in args.tasks:
        args.tasks.append("warszawski_binding_d44")
    return args


# main
def main():
    args = parse_args()
    root = repo_root()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    all_rows = []
    for task_id in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"{'='*60}")
        task_path = root / TASKS[task_id]
        frame = load_task(task_path, args.label_column, args.fold_column, args.max_rows)
        print(f"  Loaded {len(frame)} samples, {frame['cv_fold'].nunique()} folds")
        rows = run_finetuning_cv(task_id, frame, args, tokenizer, device)
        all_rows.extend(rows)

    write_outputs(
        rows=all_rows,
        output_csv=Path(args.output_csv),
        output_md=Path(args.output_md),
    )


if __name__ == "__main__":
    main()
