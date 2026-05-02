#!/usr/bin/env python3
"""Normalize Germinal outputs into Phase 3 handoff tables."""

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    yaml = None


STATUS_ORDER = {
    "accepted": 0,
    "redesign_candidate": 1,
    "trajectory": 2,
}

SCORE_COLUMNS = [
    "external_iptm",
    "i_ptm",
    "external_plddt",
    "plddt",
    "pdockq2",
    "interface_shape_comp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, object]:
    if yaml is None or not path.exists():
        return {}
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def read_csv_rows(path: Path, status: str) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = dict(row)
            row["status"] = status
            rows.append(row)
        return rows


def extract_seed(design_name: str) -> str:
    match = re.search(r"_s(\d+)", design_name)
    return match.group(1) if match else ""


def numeric_value(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def normalize_rows(
    rows: Iterable[Dict[str, str]],
    run_dir: Path,
    config_payload: Dict[str, object],
) -> Tuple[List[Dict[str, str]], List[str]]:
    run_settings = config_payload.get("run_settings", {}) if isinstance(config_payload, dict) else {}
    target_settings = config_payload.get("target_settings", {}) if isinstance(config_payload, dict) else {}
    chain_format = str(run_settings.get("type", ""))
    structure_backend = str(run_settings.get("structure_model", ""))
    target_name = str(target_settings.get("target_name", ""))

    normalized: List[Dict[str, str]] = []
    extra_columns = set()
    for row in rows:
        extra_columns.update(row.keys())
        design_name = row.get("design_name", "")
        final_structure = row.get("final_structure_path", "")
        if final_structure and not os.path.isabs(final_structure):
            candidate_path = run_dir / final_structure
            if candidate_path.exists():
                final_structure = str(candidate_path.resolve())
        normalized_row = {
            "candidate_id": design_name,
            "generator": "germinal",
            "target": target_name or row.get("target_name", ""),
            "run_id": row.get("experiment_name", run_dir.parent.name),
            "seed": extract_seed(design_name),
            "chain_format": chain_format,
            "sequence": row.get("trajectory_sequence", row.get("sequence", "")),
            "structure_backend": structure_backend,
            "status": row.get("status", ""),
            "output_structure_path": final_structure,
        }
        normalized_row.update(row)
        normalized.append(normalized_row)

    ordered_columns = [
        "candidate_id",
        "generator",
        "target",
        "run_id",
        "seed",
        "chain_format",
        "sequence",
        "structure_backend",
        "status",
        "output_structure_path",
    ]
    for column in sorted(extra_columns):
        if column not in ordered_columns:
            ordered_columns.append(column)
    return normalized, ordered_columns


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rank_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def sort_key(row: Dict[str, str]) -> Tuple[float, ...]:
        key = [STATUS_ORDER.get(row.get("status", ""), 9)]
        for column in SCORE_COLUMNS:
            key.append(-numeric_value(row.get(column, "")))
        key.append(row.get("candidate_id", ""))
        return tuple(key)

    return sorted(rows, key=sort_key)


def write_summary(
    path: Path,
    rows: List[Dict[str, str]],
    config_payload: Dict[str, object],
    run_dir: Path,
) -> None:
    run_summary_txt = run_dir / "run_summary.txt"
    counts = Counter(row.get("status", "unknown") for row in rows)
    run_settings = config_payload.get("run_settings", {}) if isinstance(config_payload, dict) else {}
    target_settings = config_payload.get("target_settings", {}) if isinstance(config_payload, dict) else {}

    lines = [
        f"# Germinal run summary: {run_dir.name}",
        "",
        f"- target: `{target_settings.get('target_name', 'unknown')}`",
        f"- chain format: `{run_settings.get('type', 'unknown')}`",
        f"- structure backend: `{run_settings.get('structure_model', 'unknown')}`",
        f"- total candidates recorded: `{len(rows)}`",
        f"- accepted: `{counts.get('accepted', 0)}`",
        f"- redesign candidates: `{counts.get('redesign_candidate', 0)}`",
        f"- trajectories only: `{counts.get('trajectory', 0)}`",
    ]
    if run_summary_txt.exists():
        lines.extend(["", "## Raw Germinal summary", "", run_summary_txt.read_text().strip()])
    path.write_text("\n".join(lines) + "\n")


def write_failure_log(path: Path, run_dir: Path, counts: Counter) -> None:
    failure_counts = run_dir / "failure_counts.csv"
    lines = [
        "No accepted Germinal designs were produced for this run.",
        "",
        f"accepted={counts.get('accepted', 0)}",
        f"redesign_candidate={counts.get('redesign_candidate', 0)}",
        f"trajectory={counts.get('trajectory', 0)}",
    ]
    if failure_counts.exists():
        lines.extend(["", "Raw failure_counts.csv:", "", failure_counts.read_text().strip()])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config_payload = load_yaml(args.run_dir / "final_config.yaml")

    staged_rows: List[Dict[str, str]] = []
    staged_rows.extend(read_csv_rows(args.run_dir / "accepted" / "designs.csv", "accepted"))
    staged_rows.extend(
        read_csv_rows(args.run_dir / "redesign_candidates" / "designs.csv", "redesign_candidate")
    )
    staged_rows.extend(read_csv_rows(args.run_dir / "trajectories" / "designs.csv", "trajectory"))

    normalized_rows, fieldnames = normalize_rows(staged_rows, args.run_dir, config_payload)
    ranked_rows = rank_rows(normalized_rows)
    top_rows = ranked_rows[:20]

    all_candidates_path = args.output_dir / f"{args.output_prefix}_all_candidates.csv"
    top_candidates_path = args.output_dir / f"{args.output_prefix}_top_candidates.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.md"
    failure_log_path = args.output_dir / f"{args.output_prefix}_failure_log.txt"

    write_csv(all_candidates_path, ranked_rows, fieldnames)
    write_csv(top_candidates_path, top_rows, fieldnames)
    write_summary(summary_path, ranked_rows, config_payload, args.run_dir)

    counts = Counter(row.get("status", "unknown") for row in ranked_rows)
    if counts.get("accepted", 0) == 0:
        write_failure_log(failure_log_path, args.run_dir, counts)
    elif failure_log_path.exists():
        failure_log_path.unlink()


if __name__ == "__main__":
    main()
