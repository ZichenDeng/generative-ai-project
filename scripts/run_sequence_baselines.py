#!/usr/bin/env python3
"""Run zero-dependency FLAb sequence baselines.

This script intentionally uses only the Python standard library so the team can
run the first milestone benchmark even before pandas, scikit-learn, PyTorch,
ESM-2, or AbMAP are configured.
"""

import argparse
import csv
import math
import os
from collections import defaultdict


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TASKS = [
    (
        "koenig_binding_g6",
        "data/processed/koenig2017mutational_kd_g6_folds.csv",
    ),
    (
        "koenig_expression_g6",
        "data/processed/koenig2017mutational_er_g6_folds.csv",
    ),
    (
        "warszawski_binding_d44",
        "data/processed/warszawski2019_d44_Kd_folds.csv",
    ),
]


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def read_rows(path, label_column, fold_column):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = parse_float(row.get(label_column))
            fold = parse_float(row.get(fold_column))
            heavy = (row.get("heavy") or "").strip()
            light = (row.get("light") or "").strip()
            if label is None or fold is None or not heavy:
                continue
            rows.append(
                {
                    "heavy": heavy,
                    "light": light,
                    "label": label,
                    "fold": int(fold),
                }
            )
    return rows


def aa_fraction_features(sequence, prefix):
    length = max(len(sequence), 1)
    counts = defaultdict(int)
    for aa in sequence:
        counts[aa] += 1
    features = []
    names = []
    for aa in AMINO_ACIDS:
        names.append("{}_frac_{}".format(prefix, aa))
        features.append(counts[aa] / float(length))
    return names, features


def grouped_fraction(sequence, group):
    if not sequence:
        return 0.0
    hits = sum(1 for aa in sequence if aa in group)
    return hits / float(len(sequence))


def sequence_features(heavy, light):
    combined = heavy + light
    names = ["heavy_len", "light_len", "total_len"]
    features = [float(len(heavy)), float(len(light)), float(len(combined))]

    for prefix, sequence in (
        ("heavy", heavy),
        ("light", light),
        ("combined", combined),
    ):
        aa_names, aa_features = aa_fraction_features(sequence, prefix)
        names.extend(aa_names)
        features.extend(aa_features)

    groups = [
        ("combined_hydrophobic_frac", "AILMFWYV"),
        ("combined_polar_frac", "STNQ"),
        ("combined_positive_frac", "KRH"),
        ("combined_negative_frac", "DE"),
        ("combined_aromatic_frac", "FWY"),
        ("combined_glycine_frac", "G"),
        ("combined_proline_frac", "P"),
        ("combined_cysteine_frac", "C"),
    ]
    for name, group in groups:
        names.append(name)
        features.append(grouped_fraction(combined, group))

    return names, features


def build_matrix(rows):
    x_rows = []
    y = []
    feature_names = None
    for row in rows:
        names, features = sequence_features(row["heavy"], row["light"])
        if feature_names is None:
            feature_names = names
        x_rows.append(features)
        y.append(row["label"])
    return feature_names or [], x_rows, y


def standardize(train_x, test_x):
    if not train_x:
        return train_x, test_x
    width = len(train_x[0])
    means = []
    stds = []
    for col in range(width):
        values = [row[col] for row in train_x]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(len(values))
        std = math.sqrt(variance)
        if std == 0.0:
            std = 1.0
        means.append(mean)
        stds.append(std)

    def transform(matrix):
        return [
            [(row[col] - means[col]) / stds[col] for col in range(width)]
            for row in matrix
        ]

    return transform(train_x), transform(test_x)


def solve_linear_system(matrix, vector):
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[pivot][col] = 1e-12
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[row][n] for row in range(n)]


def fit_ridge(train_x, train_y, alpha):
    width = len(train_x[0]) + 1
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for row, target in zip(train_x, train_y):
        with_intercept = [1.0] + row
        for i in range(width):
            xty[i] += with_intercept[i] * target
            for j in range(width):
                xtx[i][j] += with_intercept[i] * with_intercept[j]

    for i in range(1, width):
        xtx[i][i] += alpha

    return solve_linear_system(xtx, xty)


def predict_ridge(weights, x_rows):
    predictions = []
    for row in x_rows:
        total = weights[0]
        for weight, value in zip(weights[1:], row):
            total += weight * value
        predictions.append(total)
    return predictions


def mean(values):
    return sum(values) / float(len(values)) if values else float("nan")


def rmse(y_true, y_pred):
    return math.sqrt(mean([(truth - pred) ** 2 for truth, pred in zip(y_true, y_pred)]))


def mae(y_true, y_pred):
    return mean([abs(truth - pred) for truth, pred in zip(y_true, y_pred)])


def r2_score(y_true, y_pred):
    baseline = mean(y_true)
    ss_res = sum((truth - pred) ** 2 for truth, pred in zip(y_true, y_pred))
    ss_tot = sum((truth - baseline) ** 2 for truth in y_true)
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def pearson(y_true, y_pred):
    x_bar = mean(y_true)
    y_bar = mean(y_pred)
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in zip(y_true, y_pred))
    x_den = math.sqrt(sum((x - x_bar) ** 2 for x in y_true))
    y_den = math.sqrt(sum((y - y_bar) ** 2 for y in y_pred))
    denominator = x_den * y_den
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def average_ranks(values):
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    return ranks


def spearman(y_true, y_pred):
    return pearson(average_ranks(y_true), average_ranks(y_pred))


def metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def format_float(value):
    if value is None or math.isnan(value):
        return "nan"
    return "{:.6f}".format(value)


def run_task(task_id, path, label_column, fold_column, ridge_alpha):
    rows = read_rows(path, label_column, fold_column)
    folds = sorted(set(row["fold"] for row in rows))
    results = []

    for fold in folds:
        train_rows = [row for row in rows if row["fold"] != fold]
        test_rows = [row for row in rows if row["fold"] == fold]
        _, train_x, train_y = build_matrix(train_rows)
        _, test_x, test_y = build_matrix(test_rows)

        train_mean = mean(train_y)
        mean_predictions = [train_mean] * len(test_y)
        results.append(
            {
                "task": task_id,
                "model": "train_mean",
                "fold": str(fold),
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                **metrics(test_y, mean_predictions),
            }
        )

        scaled_train_x, scaled_test_x = standardize(train_x, test_x)
        weights = fit_ridge(scaled_train_x, train_y, ridge_alpha)
        ridge_predictions = predict_ridge(weights, scaled_test_x)
        results.append(
            {
                "task": task_id,
                "model": "sequence_features_ridge",
                "fold": str(fold),
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                **metrics(test_y, ridge_predictions),
            }
        )

    for model in sorted(set(row["model"] for row in results)):
        model_rows = [row for row in results if row["model"] == model]
        summary = {
            "task": task_id,
            "model": model,
            "fold": "mean",
            "n_train": sum(row["n_train"] for row in model_rows),
            "n_test": sum(row["n_test"] for row in model_rows),
        }
        for key in ("rmse", "mae", "r2", "pearson", "spearman"):
            values = [row[key] for row in model_rows if not math.isnan(row[key])]
            summary[key] = mean(values)
        results.append(summary)

    return results


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "task",
        "model",
        "fold",
        "n_train",
        "n_test",
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("rmse", "mae", "r2", "pearson", "spearman"):
                formatted[key] = format_float(formatted[key])
            writer.writerow(formatted)


def write_markdown(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary_rows = [row for row in rows if row["fold"] == "mean"]
    with open(path, "w") as handle:
        handle.write("# Baseline Results\n\n")
        handle.write(
            "These results use deterministic 10-fold splits and standard-library "
            "sequence features only. They are intended as the first milestone "
            "reference point before ESM-2 and AbMAP embeddings are added.\n\n"
        )
        handle.write(
            "| Task | Model | RMSE | MAE | R2 | Pearson | Spearman |\n"
        )
        handle.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in sorted(summary_rows, key=lambda item: (item["task"], item["model"])):
            handle.write(
                "| {task} | {model} | {rmse} | {mae} | {r2} | {pearson} | {spearman} |\n".format(
                    task=row["task"],
                    model=row["model"],
                    rmse=format_float(row["rmse"]),
                    mae=format_float(row["mae"]),
                    r2=format_float(row["r2"]),
                    pearson=format_float(row["pearson"]),
                    spearman=format_float(row["spearman"]),
                )
            )


def main():
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument("--fold-column", default="cv_fold")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(root, "results", "baseline-results.csv"),
    )
    parser.add_argument(
        "--output-md",
        default=os.path.join(root, "results", "baseline-results.md"),
    )
    args = parser.parse_args()

    all_results = []
    for task_id, relative_path in TASKS:
        path = os.path.join(root, relative_path)
        if not os.path.exists(path):
            print("Skipping missing task file: {}".format(relative_path))
            continue
        print("Running {}".format(task_id))
        all_results.extend(
            run_task(
                task_id=task_id,
                path=path,
                label_column=args.label_column,
                fold_column=args.fold_column,
                ridge_alpha=args.ridge_alpha,
            )
        )

    write_csv(args.output_csv, all_results)
    write_markdown(args.output_md, all_results)
    print("Wrote {}".format(args.output_csv))
    print("Wrote {}".format(args.output_md))


if __name__ == "__main__":
    main()
