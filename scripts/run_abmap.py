#!/usr/bin/env python3
"""Run AbMAP (ESM-2 backbone) embedding + ridge regression baselines.

AbMAP augments ESM-2 residue embeddings with CDR-focused mutagenesis, then maps
them through pretrained AbMAP attention models (see the AbMAP paper/repo).

Recommended: install the AbMAP package from its repository and dependencies
(ANARCI, dscript, etc.). This script adds the AbMAP repo root to ``sys.path``
if ``--abmap-root`` is set.

The first run may download ESM-2 via PyTorch Hub unless it is already cached.
Embeddings are cached under ``results/embeddings/``.

Multi-GPU: pass ``--gpus 0 1 2 3`` so each GPU runs a worker process that pulls one
sequence at a time from a shared task queue (``spawn``). Progress is shown with
``tqdm`` in the main process. Single-GPU behavior is unchanged if you omit ``--gpus``
or pass a single id.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

import torch.multiprocessing as torch_mp
from tqdm import tqdm


TASKS = {
    "koenig_binding_g6": "data/processed/koenig2017mutational_kd_g6_folds.csv",
    "koenig_expression_g6": "data/processed/koenig2017mutational_er_g6_folds.csv",
    "warszawski_binding_d44": "data/processed/warszawski2019_d44_Kd_folds.csv",
}

DEFAULT_ABMAP_ROOT = Path("/data/cb/scratch/ejli/abmap")
def repo_root():
    return Path(__file__).resolve().parents[1]


def cache_tag(args):
    return "abmap_esm2__k{}__{}__{}".format(
        args.abmap_k,
        args.abmap_task,
        args.abmap_plm,
    )


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


def _abmap_device_index(args):
    if args.device == "auto":
        return 0
    if args.device == "cpu":
        return 0
    if args.device.startswith("cuda:"):
        return int(args.device.split(":", 1)[1])
    return 0


def _protein_embed_device(args):
    if args.device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    return "cuda:{}".format(_abmap_device_index(args))


def _resolve_embedding_gpus(args):
    """Device indices for embedding (length 1 unless ``--gpus`` lists several)."""
    if args.gpus is not None:
        if args.device == "cpu":
            raise ValueError("--gpus is incompatible with --device cpu.")
        if not torch.cuda.is_available():
            raise ValueError("--gpus was set but CUDA is not available.")
        return list(args.gpus)
    if args.device == "cpu" or not torch.cuda.is_available():
        return [_abmap_device_index(args)]
    return [_abmap_device_index(args)]


def _use_multiprocess_embedding(gpu_ids):
    return len(gpu_ids) > 1 and torch.cuda.is_available()


def _abmap_augmented_residue_width(plm_name):
    """Last-dimension size of CDR tensors fed to ``AbMAPAttn.embed`` (PLM dim + 4 CDR labels)."""
    widths = {
        "beplerberger": 2204,
        "protbert": 1028,
        "esm1b": 1284,
        "esm2": 1284,
        "tape": 772,
    }
    if plm_name not in widths:
        raise ValueError("Unknown AbMAP PLM {!r}; expected one of {}".format(plm_name, sorted(widths)))
    return widths[plm_name]


def abmap_sequence_embedding(
    sequence,
    chain_type,
    pretrained,
    embed_device,
    k_mutations,
    task,
    plm_name,
):
    """Fixed-length AbMAP vector for one chain (empty sequence -> zeros)."""
    seq = str(sequence).strip() if sequence is not None else ""
    if not seq:
        width = _abmap_augmented_residue_width(plm_name)
        z = torch.zeros(1, 1, width, device=next(pretrained.parameters()).device)
        with torch.no_grad():
            out = pretrained.embed(z, task=task, embed_type="fixed")
        return np.zeros(out.shape[-1], dtype=np.float32)

    from abmap.abmap_augment import ProteinEmbedding

    prot = ProteinEmbedding(seq, chain_type=chain_type, embed_device=embed_device)
    cdr = prot.create_cdr_specific_embedding(
        embed_type=plm_name,
        k=k_mutations,
        separator=False,
        mask=True,
    )
    batch = torch.unsqueeze(cdr, dim=0)
    dev = next(pretrained.parameters()).device
    batch = batch.to(dev)
    with torch.no_grad():
        out = pretrained.embed(batch, task=task, embed_type="fixed")
    vec = torch.squeeze(out, dim=0).detach().cpu().numpy().astype(np.float32)
    return vec


def embed_sequences_abmap(
    sequences,
    chain_type,
    pretrained,
    embed_device,
    k_mutations,
    task,
    plm_name,
):
    embeddings = {}
    desc = "AbMAP {} chain".format(chain_type)
    for sequence in tqdm(sequences, desc=desc):
        embeddings[sequence] = abmap_sequence_embedding(
            sequence,
            chain_type=chain_type,
            pretrained=pretrained,
            embed_device=embed_device,
            k_mutations=k_mutations,
            task=task,
            plm_name=plm_name,
        )
    return embeddings


def _gpu_sequence_consumer(
    gpu_id,
    task_queue,
    result_queue,
    chain_type,
    pretrained_path,
    plm_name,
    abmap_task,
    abmap_k,
):
    """One process per GPU: load AbMAP once, then embed single sequences from *task_queue*."""

    import torch as th

    th.cuda.set_device(gpu_id)
    from abmap.commands.embed import load_abmap

    pretrained = load_abmap(pretrained_path, plm_name, gpu_id)
    embed_device = "cuda:{}".format(gpu_id)

    while True:
        sequence = task_queue.get()
        if sequence is None:
            break
        vec = abmap_sequence_embedding(
            sequence,
            chain_type=chain_type,
            pretrained=pretrained,
            embed_device=embed_device,
            k_mutations=abmap_k,
            task=abmap_task,
            plm_name=plm_name,
        )
        result_queue.put((sequence, vec))


def _embed_sequences_abmap_distributed(
    sequences,
    chain_type,
    pretrained_path,
    args,
    gpu_ids,
):
    """One consumer process per GPU; tasks are single sequences (dynamic load balancing)."""
    if not sequences:
        return {}

    seq_list = sorted(sequences)
    ctx = torch_mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    procs = []
    for gpu_id in gpu_ids:
        proc = ctx.Process(
            target=_gpu_sequence_consumer,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                chain_type,
                str(pretrained_path),
                args.abmap_plm,
                args.abmap_task,
                args.abmap_k,
            ),
        )
        proc.start()
        procs.append(proc)

    for seq in seq_list:
        task_queue.put(seq)
    for _ in gpu_ids:
        task_queue.put(None)

    merged = {}
    desc = "AbMAP {} chain".format(chain_type)
    for _ in tqdm(range(len(seq_list)), desc=desc):
        sequence, vec = result_queue.get()
        merged[sequence] = vec

    for proc in procs:
        proc.join()
    return merged


def load_or_create_embeddings(
    frame,
    task_id,    
    args,
    abmap_h,
    abmap_l,
    embed_device,
    gpu_ids,
):
    root = repo_root()
    cache_dir = root / "results" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "{}__{}.npz".format(cache_tag(args), task_id)

    if cache_path.exists() and not args.force_embeddings:
        cached = np.load(cache_path, allow_pickle=True)
        if cached["x"].shape[0] == len(frame):
            return cached["x"]
        print("Ignoring embedding cache with mismatched row count: {}".format(cache_path))

    heavy_seqs = sorted(
        {s for s in frame["heavy"].astype(str).tolist() if s and str(s).strip()}
    )
    light_seqs = sorted(
        {s for s in frame["light"].astype(str).tolist() if s and str(s).strip()}
    )

    task = args.abmap_task
    k = args.abmap_k

    plm = args.abmap_plm

    if _use_multiprocess_embedding(gpu_ids):
        heavy_embeddings = _embed_sequences_abmap_distributed(
            heavy_seqs,
            "H",
            args.pretrained_heavy,
            args,
            gpu_ids,
        )
        light_embeddings = _embed_sequences_abmap_distributed(
            light_seqs,
            "L",
            args.pretrained_light,
            args,
            gpu_ids,
        )
    else:
        heavy_embeddings = embed_sequences_abmap(
            heavy_seqs,
            chain_type="H",
            pretrained=abmap_h,
            embed_device=embed_device,
            k_mutations=k,
            task=task,
            plm_name=plm,
        )
        light_embeddings = embed_sequences_abmap(
            light_seqs,
            chain_type="L",
            pretrained=abmap_l,
            embed_device=embed_device,
            k_mutations=k,
            task=task,
            plm_name=plm,
        )

    if heavy_embeddings:
        zero_h = np.zeros_like(next(iter(heavy_embeddings.values())))
    else:
        zero_h = None
    if light_embeddings:
        zero_l = np.zeros_like(next(iter(light_embeddings.values())))
    else:
        zero_l = None
    if zero_h is None and zero_l is None:
        raise ValueError("No non-empty heavy or light sequences")
    if zero_h is None:
        zero_h = np.zeros_like(zero_l)
    if zero_l is None:
        zero_l = np.zeros_like(zero_h)

    x_rows = []
    for _, row in frame.iterrows():
        heavy = heavy_embeddings.get(str(row["heavy"]).strip(), zero_h)
        light = light_embeddings.get(str(row["light"]).strip(), zero_l)
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
                "model": "abmap_esm2_ridge",
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
        "model": "abmap_esm2_ridge",
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
        handle.write("# AbMAP (ESM-2) Ridge Results\n\n")
        handle.write(
            "These results use pretrained AbMAP embeddings (ESM-2 backbone, CDR augmentation) "
            "and a ridge regression head.\n\n"
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
    parser.add_argument(
        "--abmap-root",
        type=Path,
        default=DEFAULT_ABMAP_ROOT,
        help="Path to the AbMAP repository (added to sys.path)",
    )
    parser.add_argument(
        "--pretrained-heavy",
        type=Path,
        default=None,
        help="AbMAP heavy-chain checkpoint (default: <abmap-root>/pretrained_models/AbMAP_esm2_H.pt)",
    )
    parser.add_argument(
        "--pretrained-light",
        type=Path,
        default=None,
        help="AbMAP light-chain checkpoint (default: <abmap-root>/pretrained_models/AbMAP_esm2_L.pt)",
    )
    parser.add_argument(
        "--abmap-plm",
        default="esm2",
        help="Foundational PLM name passed to AbMAP (must match checkpoints; default esm2)",
    )
    parser.add_argument(
        "--abmap-task",
        choices=("structure", "function"),
        default="function",
        help="AbMAP embed() task: use 'function' for affinity/expression-style targets (default)",
    )
    parser.add_argument(
        "--abmap-k",
        type=int,
        default=100,
        help="Mutagenesis samples for CDR augmentation (default 100; lower is faster)",
    )
    parser.add_argument("--tasks", nargs="+", default=["koenig_binding_g6", "koenig_expression_g6"])
    parser.add_argument("--include-warszawski", action="store_true")
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument("--fold-column", default="cv_fold")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="CUDA device indices for parallel embedding (e.g. 0 1 2 3). "
        "Each GPU runs a separate process (spawn). Incompatible with --device cpu.",
    )
    parser.add_argument("--max-rows-per-task", type=int)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument(
        "--output-csv",
        default=str(root / "results" / "abmap-esm2-results.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=str(root / "results" / "abmap-esm2-results.md"),
    )
    args = parser.parse_args()
    pm = args.abmap_root / "pretrained_models"
    if args.pretrained_heavy is None:
        args.pretrained_heavy = pm / "AbMAP_{}_H.pt".format(args.abmap_plm)
    if args.pretrained_light is None:
        args.pretrained_light = pm / "AbMAP_{}_L.pt".format(args.abmap_plm)
    if args.include_warszawski and "warszawski_binding_d44" not in args.tasks:
        args.tasks.append("warszawski_binding_d44")
    return args


def main():
    args = parse_args()

    gpu_ids = _resolve_embedding_gpus(args)
    if _use_multiprocess_embedding(gpu_ids):
        n_dev = torch.cuda.device_count()
        for g in gpu_ids:
            if g < 0 or g >= n_dev:
                raise ValueError(
                    "GPU index {} is out of range ({} device(s) visible)".format(g, n_dev)
                )
        print("Multi-GPU embedding on devices: {}".format(gpu_ids))

    from abmap.commands.embed import load_abmap

    root = repo_root()
    dev_idx = _abmap_device_index(args)
    embed_device = _protein_embed_device(args)

    if not args.pretrained_heavy.is_file():
        raise FileNotFoundError(
            "Missing AbMAP heavy checkpoint: {} (set --pretrained-heavy or --abmap-root)".format(
                args.pretrained_heavy
            )
        )
    if not args.pretrained_light.is_file():
        raise FileNotFoundError(
            "Missing AbMAP light checkpoint: {} (set --pretrained-light or --abmap-root)".format(
                args.pretrained_light
            )
        )

    multi = _use_multiprocess_embedding(gpu_ids)
    if multi:
        abmap_h = None
        abmap_l = None
    else:
        print("Loading AbMAP ({}): {}".format(args.abmap_plm, args.pretrained_heavy.name))
        abmap_h = load_abmap(
            str(args.pretrained_heavy),
            args.abmap_plm,
            dev_idx,
        )
        print("Loading AbMAP ({}): {}".format(args.abmap_plm, args.pretrained_light.name))
        abmap_l = load_abmap(
            str(args.pretrained_light),
            args.abmap_plm,
            dev_idx,
        )

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
        x = load_or_create_embeddings(
            frame,
            task_id,
            args,
            abmap_h,
            abmap_l,
            embed_device,
            gpu_ids,
        )
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
