#!/usr/bin/env python3

import os
import re
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification


AA = set("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_METRICS = {
    "binding_koenig": "koenig_binding_g6/fold0_best.pt",
    "expression_koenig": "koenig_expression_g6/fold0_best.pt",
    "binding_warszawski": "warszawski_binding_d44/fold0_best.pt",
}


def connect_huggingface(token=None):
    """
    Connect to Hugging Face.

    Use one of:
    1. token argument
    2. HF_TOKEN environment variable
    3. already logged-in Hugging Face session
    """
    token = token or os.environ.get("HF_TOKEN")

    if token:
        login(token=token)
        print("Connected to Hugging Face with token.")
    else:
        print("No HF token provided. Using public access or existing login.")


def clean_sequence(seq):
    seq = str(seq).upper()
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq


def pick_sequence_column(df):
    for col in ["sequence", "trajectory_sequence", "binder_sequence", "seq"]:
        if col in df.columns:
            return col

    raise ValueError(f"No sequence column found. Columns: {list(df.columns)}")


def parse_metrics(metrics_string):
    if metrics_string is None:
        return DEFAULT_METRICS

    metrics = {}
    for item in metrics_string.split(","):
        name, ckpt = item.split("=", 1)
        metrics[name.strip()] = ckpt.strip()

    return metrics


def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-8)


def load_metric_model(base_model, hf_repo, ckpt_file, device):
    """
    Download checkpoint from Hugging Face and load into ESM2 regression model.
    """

    ckpt_path = hf_hub_download(
        repo_id=hf_repo,
        filename=ckpt_file,
        repo_type="model",
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    remapped = {}
    for key, value in state_dict.items():
        new_key = key

        # Your checkpoint uses esm2.* but Hugging Face classifier expects esm.*
        if new_key.startswith("esm2."):
            new_key = new_key.replace("esm2.", "esm.", 1)

        remapped[new_key] = value

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        problem_type="regression",
    )

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    print(f"\nLoaded: {ckpt_file}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def score_sequences(model, tokenizer, sequences, device, batch_size=4):
    scores = []

    for start in tqdm(range(0, len(sequences), batch_size), desc="Scoring"):
        batch = sequences[start:start + batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=1022,
            return_tensors="pt",
        ).to(device)

        output = model(**tokens)
        batch_scores = output.logits.squeeze(-1).detach().cpu().float().numpy()
        scores.extend(batch_scores.tolist())

    return np.array(scores)


def run_pipeline(args):
    connect_huggingface(args.hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    seq_col = args.sequence_column or pick_sequence_column(df)
    print("Using sequence column:", seq_col)

    df["phase3_sequence"] = df[seq_col].map(clean_sequence)

    before = len(df)
    df = df[df["phase3_sequence"].str.len().between(args.min_len, args.max_len)].copy()
    df = df[df["phase3_sequence"].map(lambda s: set(s).issubset(AA))].copy()

    if args.drop_duplicates:
        df = df.drop_duplicates(subset=["phase3_sequence"]).copy()

    print(f"QC kept {len(df)} / {before} candidates")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    metrics = parse_metrics(args.metrics)
    sequences = df["phase3_sequence"].tolist()

    for metric_name, ckpt_file in metrics.items():
        print(f"\nScoring metric: {metric_name}")

        model = load_metric_model(
            base_model=args.base_model,
            hf_repo=args.hf_repo,
            ckpt_file=ckpt_file,
            device=device,
        )

        df[f"esm2_{metric_name}"] = score_sequences(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            device=device,
            batch_size=args.batch_size,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_cols = [c for c in df.columns if c.startswith("esm2_")]

    for col in score_cols:
        df[col + "_z"] = zscore(df[col])

    z_cols = [c for c in df.columns if c.endswith("_z")]
    df["phase3_composite_score"] = df[z_cols].mean(axis=1)

    df = df.sort_values("phase3_composite_score", ascending=False).reset_index(drop=True)
    df["phase3_rank"] = np.arange(1, len(df) + 1)

    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print(df[["phase3_rank", "phase3_composite_score", "phase3_sequence"] + score_cols].head(10))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase 3 Germinal candidate reranking with Hugging Face ESM2 models"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--hf_repo",
        default="zhangtin/antibody-esm2-finetuned",
        help="Hugging Face repo containing fine-tuned ESM2 checkpoints",
    )

    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional Hugging Face token. Can also use HF_TOKEN environment variable.",
    )

    parser.add_argument(
        "--base_model",
        default="facebook/esm2_t33_650M_UR50D",
        help="Base ESM2 model matching fine-tuned checkpoint hidden size.",
    )

    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric checkpoint list: metric=path.pt,metric2=path2.pt",
    )

    parser.add_argument("--sequence_column", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_len", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=1022)
    parser.add_argument("--drop_duplicates", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)