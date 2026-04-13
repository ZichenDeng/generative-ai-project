#!/usr/bin/env python3
"""
Run small MLP benchmarks on frozen ESM-2 and AbMAP embeddings.

This script:
1. Downloads pinned embedding .npz files and processed fold CSVs from the repo
   commit if they are not already present locally.
2. Evaluates single-task MLP models for each task x embedding family.
3. Evaluates shared-task MLP models for each embedding family.
4. Saves fold-level results, out-of-fold predictions, and compact summaries.

Example:
    python scripts/run_mlp_benchmarks.py

Recommended repo structure:
    repo/
      data/processed/
      results/embeddings/
      results/mlp_benchmarks/
      scripts/run_mlp_benchmarks.py
"""

from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


COMMIT = "ac5c2475ed0aa534ddf1aa852f60c249d590b198"
RAW_BASE = f"https://raw.githubusercontent.com/ZichenDeng/generative-ai-project/{COMMIT}"

EMBED_FILES = {
    "esm2_koenig_binding": "results/embeddings/facebook__esm2_t6_8M_UR50D__koenig_binding_g6.npz",
    "esm2_koenig_expression": "results/embeddings/facebook__esm2_t6_8M_UR50D__koenig_expression_g6.npz",
    "esm2_warszawski_binding": "results/embeddings/facebook__esm2_t6_8M_UR50D__warszawski_binding_d44.npz",
    "abmap_koenig_binding": "results/embeddings/abmap_esm2__k100__function__esm2__koenig_binding_g6.npz",
    "abmap_koenig_expression": "results/embeddings/abmap_esm2__k100__function__esm2__koenig_expression_g6.npz",
    "abmap_warszawski_binding": "results/embeddings/abmap_esm2__k100__function__esm2__warszawski_binding_d44.npz",
}

CSV_FILES = {
    "koenig_binding": "data/processed/koenig2017mutational_kd_g6_folds.csv",
    "koenig_expression": "data/processed/koenig2017mutational_er_g6_folds.csv",
    "warszawski_binding": "data/processed/warszawski2019_d44_Kd_folds.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-task and shared-task MLP benchmarks.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <repo-root>/results/mlp_benchmarks",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 64],
        help="Hidden dimensions for the MLP.",
    )
    parser.add_argument(
        "--task-emb-dim",
        type=int,
        default=16,
        help="Task embedding dimension for shared-task MLP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cpu or cuda. Defaults to auto-detect.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_override: str | None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] {dst}")
        return
    print(f"[download] {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def ensure_inputs(repo_root: Path) -> tuple[Path, Path]:
    emb_dir = repo_root / "results" / "embeddings"
    data_dir = repo_root / "data" / "processed"

    for relpath in EMBED_FILES.values():
        dst = repo_root / relpath
        download(f"{RAW_BASE}/{relpath}", dst)

    for relpath in CSV_FILES.values():
        dst = repo_root / relpath
        download(f"{RAW_BASE}/{relpath}", dst)

    print("\nEmbeddings:")
    for p in sorted(emb_dir.glob("*.npz")):
        print("-", p.name)

    print("\nCSV files:")
    for p in sorted(data_dir.glob("*.csv")):
        print("-", p.name)

    return emb_dir, data_dir


def build_task_map(repo_root: Path) -> dict[str, dict[str, Path]]:
    emb_dir = repo_root / "results" / "embeddings"
    data_dir = repo_root / "data" / "processed"
    return {
        "koenig_binding": {
            "csv": data_dir / "koenig2017mutational_kd_g6_folds.csv",
            "esm2": emb_dir / "facebook__esm2_t6_8M_UR50D__koenig_binding_g6.npz",
            "abmap": emb_dir / "abmap_esm2__k100__function__esm2__koenig_binding_g6.npz",
        },
        "koenig_expression": {
            "csv": data_dir / "koenig2017mutational_er_g6_folds.csv",
            "esm2": emb_dir / "facebook__esm2_t6_8M_UR50D__koenig_expression_g6.npz",
            "abmap": emb_dir / "abmap_esm2__k100__function__esm2__koenig_expression_g6.npz",
        },
        "warszawski_binding": {
            "csv": data_dir / "warszawski2019_d44_Kd_folds.csv",
            "esm2": emb_dir / "facebook__esm2_t6_8M_UR50D__warszawski_binding_d44.npz",
            "abmap": emb_dir / "abmap_esm2__k100__function__esm2__warszawski_binding_d44.npz",
        },
    }


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    val = spearmanr(y_true, y_pred).correlation
    return float(val) if val is not None and np.isfinite(val) else np.nan


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        val = pearsonr(y_true, y_pred)[0]
        return float(val) if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y_true)),
        "spearman": safe_spearman(y_true, y_pred),
        "pearson": safe_pearson(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def normalize_folds(fold: np.ndarray) -> np.ndarray:
    uniq = sorted(pd.unique(fold))
    mapping = {u: i + 1 for i, u in enumerate(uniq)}
    return np.array([mapping[v] for v in fold], dtype=int)


def load_task_data(task_map: dict[str, dict[str, Path]], task_name: str, family: str) -> dict[str, Any]:
    cfg = task_map[task_name]
    csv_path = cfg["csv"]
    npz_path = cfg[family]

    df = pd.read_csv(csv_path).copy()
    arr = np.load(npz_path, allow_pickle=True)

    if "x" not in arr:
        raise ValueError(f"{npz_path.name}: expected key 'x', found {list(arr.keys())}")

    X = np.asarray(arr["x"], dtype=np.float32)

    required_cols = ["fitness", "cv_fold"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns {missing}. Found {list(df.columns)}")

    keep = df["fitness"].notna() & df["cv_fold"].notna()
    df = df.loc[keep].reset_index(drop=True)

    if len(df) != len(X):
        if "keep_idx" in arr:
            keep_idx = np.asarray(arr["keep_idx"]).reshape(-1)
            df = pd.read_csv(csv_path).iloc[keep_idx].reset_index(drop=True)
            df = df[df["fitness"].notna() & df["cv_fold"].notna()].reset_index(drop=True)

        if len(df) != len(X):
            raise ValueError(
                f"Length mismatch for {task_name}-{family}: len(df)={len(df)} vs len(X)={len(X)}"
            )

    y = df["fitness"].to_numpy(np.float32)
    fold = df["cv_fold"].to_numpy()

    return {
        "X": X,
        "y": y,
        "fold": fold,
        "df": df,
        "task": task_name,
        "family": family,
        "embed_file": npz_path.name,
        "csv_file": csv_path.name,
    }


class RegDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class SharedTaskDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, task_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i], self.task_ids[i]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dims: tuple[int, ...] = (256, 64),
        task_emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks, task_emb_dim)

        layers: list[nn.Module] = []
        d = input_dim + task_emb_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        te = self.task_embedding(task_ids)
        z = torch.cat([x, te], dim=1)
        h = self.trunk(z)

        out = torch.zeros((x.size(0), 1), device=x.device)
        for t in range(len(self.heads)):
            mask = task_ids == t
            if mask.any():
                out[mask] = self.heads[t](h[mask])
        return out


def fit_single_fold_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    hidden_dims: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_loader = DataLoader(RegDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RegDataset(X_val_s, y_val), batch_size=batch_size, shuffle=False)

    model = MLPRegressor(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        mean_val = float(np.mean(val_losses))
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was saved.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_val_s, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)
    return pred


def evaluate_single_task(
    task_map: dict[str, dict[str, Path]],
    task_name: str,
    family: str,
    *,
    device: torch.device,
    hidden_dims: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obj = load_task_data(task_map, task_name, family)
    X, y, fold = obj["X"], obj["y"], normalize_folds(obj["fold"])

    rows = []
    oof_pred = np.full(len(y), np.nan, dtype=np.float32)

    for f in sorted(np.unique(fold)):
        tr = fold != f
        va = fold == f

        pred = fit_single_fold_mlp(
            X[tr],
            y[tr],
            X[va],
            y[va],
            device=device,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
        )
        oof_pred[va] = pred

        m = regression_metrics(y[va], pred)
        m.update({"task": task_name, "family": family, "fold": int(f), "model": "single_task_mlp"})
        rows.append(m)

    overall = regression_metrics(y, oof_pred)
    overall.update({"task": task_name, "family": family, "fold": "oof_all", "model": "single_task_mlp"})
    rows.append(overall)

    pred_df = pd.DataFrame(
        {
            "task": task_name,
            "family": family,
            "model": "single_task_mlp",
            "y_true": y,
            "y_pred": oof_pred,
            "fold": fold,
        }
    )
    return pd.DataFrame(rows), pred_df


def build_shared_family_dataset(
    task_map: dict[str, dict[str, Path]],
    family: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    blocks = []
    task_to_id: dict[str, int] = {}

    for task_name in task_map:
        obj = load_task_data(task_map, task_name, family)
        if task_name not in task_to_id:
            task_to_id[task_name] = len(task_to_id)

        df = pd.DataFrame(
            {
                "task": task_name,
                "task_id": task_to_id[task_name],
                "fold": normalize_folds(obj["fold"]),
                "y": obj["y"],
            }
        )
        blocks.append((df, obj["X"]))

    dims = [X.shape[1] for _, X in blocks]
    if len(set(dims)) != 1:
        raise ValueError(f"{family} feature dimensions differ across tasks: {dims}")

    df_all = pd.concat([b[0] for b in blocks], ignore_index=True)
    X_all = np.concatenate([b[1] for b in blocks], axis=0)
    return df_all, X_all, task_to_id


def fit_shared_fold_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    t_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    t_val: np.ndarray,
    *,
    device: torch.device,
    num_tasks: int,
    hidden_dims: tuple[int, ...],
    task_emb_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_loader = DataLoader(
        SharedTaskDataset(X_train_s, y_train, t_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SharedTaskDataset(X_val_s, y_val, t_val),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SharedTaskMLP(
        input_dim=X_train.shape[1],
        num_tasks=num_tasks,
        hidden_dims=hidden_dims,
        task_emb_dim=task_emb_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb, tb in train_loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            optimizer.zero_grad()
            pred = model(xb, tb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, tb in val_loader:
                xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
                pred = model(xb, tb)
                val_losses.append(criterion(pred, yb).item())

        mean_val = float(np.mean(val_losses))
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was saved.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = model(
            torch.tensor(X_val_s, dtype=torch.float32).to(device),
            torch.tensor(t_val, dtype=torch.long).to(device),
        ).cpu().numpy().reshape(-1)

    return pred


def evaluate_shared_family(
    task_map: dict[str, dict[str, Path]],
    family: str,
    *,
    device: torch.device,
    hidden_dims: tuple[int, ...],
    task_emb_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, X, task_to_id = build_shared_family_dataset(task_map, family)

    y = df["y"].to_numpy(np.float32)
    fold = df["fold"].to_numpy(int)
    task_ids = df["task_id"].to_numpy(int)

    oof_pred = np.full(len(y), np.nan, dtype=np.float32)

    for f in sorted(np.unique(fold)):
        tr = fold != f
        va = fold == f

        pred = fit_shared_fold_mlp(
            X[tr],
            y[tr],
            task_ids[tr],
            X[va],
            y[va],
            task_ids[va],
            device=device,
            num_tasks=len(task_to_id),
            hidden_dims=hidden_dims,
            task_emb_dim=task_emb_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
        )
        oof_pred[va] = pred

    inv_task = {v: k for k, v in task_to_id.items()}
    rows = []

    for task_id, task_name in sorted(inv_task.items()):
        mask = task_ids == task_id
        m = regression_metrics(y[mask], oof_pred[mask])
        m.update({"family": family, "task": task_name, "model": "shared_task_mlp"})
        rows.append(m)

    overall = regression_metrics(y, oof_pred)
    overall.update({"family": family, "task": "all_tasks_combined", "model": "shared_task_mlp"})
    rows.append(overall)

    pred_df = df.copy()
    pred_df["family"] = family
    pred_df["model"] = "shared_task_mlp"
    pred_df["y_pred"] = oof_pred
    pred_df["y_true"] = pred_df["y"]
    return pd.DataFrame(rows), pred_df


def save_markdown_summary(df: pd.DataFrame, path: Path, title: str) -> None:
    out = df.copy()
    for col in ["spearman", "pearson", "mae", "rmse", "r2"]:
        if col in out.columns:
            out[col] = out[col].astype(float).round(4)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        try:
            f.write(out.to_markdown(index=False))
        except Exception:
            f.write(out.to_string(index=False))
            f.write("\n")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (repo_root / "results" / "mlp_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)
    hidden_dims = tuple(args.hidden_dims)

    print(f"Repo root: {repo_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Hidden dims: {hidden_dims}")

    ensure_inputs(repo_root)
    task_map = build_task_map(repo_root)

    print("\nSanity check:")
    for task_name in task_map:
        for family in ["esm2", "abmap"]:
            obj = load_task_data(task_map, task_name, family)
            print(task_name, family, obj["X"].shape, obj["y"].shape, pd.Series(obj["fold"]).nunique())

    single_results = []
    single_preds = []

    for task_name in task_map:
        for family in ["esm2", "abmap"]:
            print(f"\nRunning single-task MLP: {task_name} | {family}")
            res, pred = evaluate_single_task(
                task_map,
                task_name,
                family,
                device=device,
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
            )
            print(res)
            single_results.append(res)
            single_preds.append(pred)

    single_results_df = pd.concat(single_results, ignore_index=True)
    single_preds_df = pd.concat(single_preds, ignore_index=True)

    single_results_df.to_csv(output_dir / "single_task_mlp_results.csv", index=False)
    single_preds_df.to_csv(output_dir / "single_task_mlp_predictions.csv", index=False)

    single_summary = (
        single_results_df[single_results_df["fold"].astype(str) == "oof_all"]
        [["task", "family", "model", "spearman", "pearson", "mae", "rmse", "r2", "n"]]
        .sort_values(["task", "family"])
        .reset_index(drop=True)
    )
    single_summary.to_csv(output_dir / "single_task_mlp_summary.csv", index=False)
    save_markdown_summary(single_summary, output_dir / "single_task_mlp_summary.md", "Single-task MLP Summary")

    shared_results = []
    shared_preds = []

    for family in ["esm2", "abmap"]:
        print(f"\nRunning shared-task MLP: {family}")
        res, pred = evaluate_shared_family(
            task_map,
            family,
            device=device,
            hidden_dims=hidden_dims,
            task_emb_dim=args.task_emb_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
        )
        print(res)
        shared_results.append(res)
        shared_preds.append(pred)

    shared_results_df = pd.concat(shared_results, ignore_index=True)
    shared_preds_df = pd.concat(shared_preds, ignore_index=True)

    shared_results_df.to_csv(output_dir / "shared_task_mlp_results.csv", index=False)
    shared_preds_df.to_csv(output_dir / "shared_task_mlp_predictions.csv", index=False)

    shared_summary = (
        shared_results_df
        [["task", "family", "model", "spearman", "pearson", "mae", "rmse", "r2", "n"]]
        .sort_values(["family", "task"])
        .reset_index(drop=True)
    )
    shared_summary.to_csv(output_dir / "shared_task_mlp_summary.csv", index=False)
    save_markdown_summary(shared_summary, output_dir / "shared_task_mlp_summary.md", "Shared-task MLP Summary")

    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# MLP Benchmark Outputs\n\n")
        f.write("- `single_task_mlp_results.csv`: fold-level single-task results\n")
        f.write("- `single_task_mlp_predictions.csv`: out-of-fold single-task predictions\n")
        f.write("- `single_task_mlp_summary.csv`: compact single-task summary\n")
        f.write("- `shared_task_mlp_results.csv`: shared-task family results\n")
        f.write("- `shared_task_mlp_predictions.csv`: shared-task predictions\n")
        f.write("- `shared_task_mlp_summary.csv`: compact shared-task summary\n")

    print("\nSaved outputs:")
    for p in sorted(output_dir.iterdir()):
        print("-", p.name)


if __name__ == "__main__":
    main()