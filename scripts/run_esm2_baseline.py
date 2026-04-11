#!/usr/bin/env python3
"""Run ESM-2 embedding + ridge regression baselines.

Recommended environment on the current machine:

    /home/zichende/.conda/envs/afffood/bin/python scripts/run_esm2_baseline.py

The first run downloads the Hugging Face ESM-2 model unless it is already
cached. Embeddings are cached under `results/embeddings/`.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, EsmModel


TASKS = {
    "koenig_binding_g6": "data/processed/koenig2017mutational_kd_g6_folds.csv",
    "koenig_expression_g6": "data/processed/koenig2017mutational_er_g6_folds.csv",
    "warszawski_binding_d44": "data/processed/warszawski2019_d44_Kd_folds.csv",
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def safe_model_name(model_name):
    return model_name.replace("/", "__").replace("-", "_")


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


def mean_pool(last_hidden_state, attention_mask, special_tokens_mask):
    valid_mask = attention_mask.bool() & ~special_tokens_mask.bool()
    valid_mask = valid_mask.unsqueeze(-1)
    summed = (last_hidden_state * valid_mask).sum(dim=1)
    counts = valid_mask.sum(dim=1).clamp(min=1)
    return summed / counts


def embed_sequences(sequences, tokenizer, model, device, batch_size):
    embeddings = {}
    model.eval()
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        pooled = mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"],
            encoded["special_tokens_mask"],
        )
        pooled = pooled.detach().cpu().numpy()
        for sequence, vector in zip(batch, pooled):
            embeddings[sequence] = vector
    return embeddings


def load_or_create_embeddings(frame, task_id, args, tokenizer, model, device):
    root = repo_root()
    cache_dir = root / "results" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "{}__{}.npz".format(
        safe_model_name(args.model_name),
        task_id,
    )

    if cache_path.exists() and not args.force_embeddings:
        cached = np.load(cache_path, allow_pickle=True)
        if cached["x"].shape[0] == len(frame):
            return cached["x"]
        print("Ignoring embedding cache with mismatched row count: {}".format(cache_path))

    sequences = sorted(
        set(frame["heavy"].astype(str).tolist())
        | set(frame["light"].astype(str).tolist())
    )
    sequences = [sequence for sequence in sequences if sequence]
    sequence_embeddings = embed_sequences(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    zero = np.zeros(next(iter(sequence_embeddings.values())).shape, dtype=np.float32)
    x_rows = []
    for _, row in frame.iterrows():
        heavy = sequence_embeddings.get(str(row["heavy"]), zero)
        light = sequence_embeddings.get(str(row["light"]), zero)
        x_rows.append(np.concatenate([heavy, light]))
    x = np.vstack(x_rows)
    np.savez_compressed(cache_path, x=x)
    return x


def correlations(y_true, y_pred):
    if len(np.unique(y_pred)) <= 1 or len(np.unique(y_true)) <= 1:
        return np.nan, np.nan
    return pearsonr(y_true, y_pred).statistic, spearmanr(y_true, y_pred).statistic


def run_ridge_cv(task_id, frame, x, label_column, fold_column, alpha):
    y = frame[label_column].to_numpy(dtype=float)
    folds = sorted(frame[fold_column].unique())
    rows = []
    for fold in folds:
        train_mask = frame[fold_column].to_numpy() != fold
        test_mask = ~train_mask

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_mask])
        x_test = scaler.transform(x[test_mask])
        y_train = y[train_mask]
        y_test = y[test_mask]

        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        pearson, spearman = correlations(y_test, y_pred)

        rows.append(
            {
                "task": task_id,
                "model": "esm2_ridge",
                "fold": fold,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "pearson": pearson,
                "spearman": spearman,
            }
        )

    mean_row = {
        "task": task_id,
        "model": "esm2_ridge",
        "fold": "mean",
        "n_train": int(sum(row["n_train"] for row in rows)),
        "n_test": int(sum(row["n_test"] for row in rows)),
    }
    for metric in ("rmse", "mae", "r2", "pearson", "spearman"):
        mean_row[metric] = float(np.nanmean([row[metric] for row in rows]))
    rows.append(mean_row)
    return rows


def write_outputs(rows, output_csv, output_md):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)

    summary = frame[frame["fold"].astype(str) == "mean"].copy()
    with open(output_md, "w") as handle:
        handle.write("# ESM-2 Ridge Results\n\n")
        handle.write(
            "These results use frozen ESM-2 embeddings and a ridge regression head.\n\n"
        )
        handle.write(
            "| Task | Model | RMSE | MAE | R2 | Pearson | Spearman |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for _, row in summary.sort_values(["task", "model"]).iterrows():
            handle.write(
                "| {task} | {model} | {rmse:.6f} | {mae:.6f} | {r2:.6f} | {pearson:.6f} | {spearman:.6f} |\n".format(
                    task=row["task"],
                    model=row["model"],
                    rmse=row["rmse"],
                    mae=row["mae"],
                    r2=row["r2"],
                    pearson=row["pearson"],
                    spearman=row["spearman"],
                )
            )


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--tasks", nargs="+", default=["koenig_binding_g6", "koenig_expression_g6"])
    parser.add_argument("--include-warszawski", action="store_true")
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument("--fold-column", default="cv_fold")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows-per-task", type=int)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-csv", default=str(root / "results" / "esm2-baseline-results.csv"))
    parser.add_argument("--output-md", default=str(root / "results" / "esm2-baseline-results.md"))
    args = parser.parse_args()
    if args.include_warszawski and "warszawski_binding_d44" not in args.tasks:
        args.tasks.append("warszawski_binding_d44")
    return args


def main():
    args = parse_args()
    root = repo_root()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Loading {}".format(args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = EsmModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    ).to(device)

    all_rows = []
    for task_id in args.tasks:
        task_path = root / TASKS[task_id]
        print("Running {}".format(task_id))
        frame = load_task(
            task_path,
            label_column=args.label_column,
            fold_column=args.fold_column,
            max_rows=args.max_rows_per_task,
        )
        x = load_or_create_embeddings(frame, task_id, args, tokenizer, model, device)
        all_rows.extend(
            run_ridge_cv(
                task_id=task_id,
                frame=frame,
                x=x,
                label_column=args.label_column,
                fold_column=args.fold_column,
                alpha=args.ridge_alpha,
            )
        )

    write_outputs(
        rows=all_rows,
        output_csv=Path(args.output_csv),
        output_md=Path(args.output_md),
    )
    print("Wrote {}".format(args.output_csv))
    print("Wrote {}".format(args.output_md))


if __name__ == "__main__":
    main()
