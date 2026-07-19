#!/usr/bin/env python3
"""Combine MindNav val_mini (30) and val_60 (60) by KG format."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_valmini_kg_legacy import (
    audit_run as audit_legacy_run,
    load_episode_manifest,
    load_jsonl,
    mapping_diagnostics,
    percentile_ci,
    write_csv,
)


FORMATS = ("text", "json", "triples")
SPLITS = ("val_mini", "val_60")
METRICS = ("habitat_success", "habitat_spl", "distance_to_goal", "steps")


def audit_val60(
    label: str,
    path: Path,
    records: list[dict],
    expected_n: int,
    success_threshold: float,
) -> list[dict]:
    if len(records) != expected_n:
        raise ValueError(f"{path}: N={len(records)}, expected {expected_n}")
    expected_serialization = label
    manifest = []
    for index, record in enumerate(records):
        if int(record.get("episode_index", -1)) != index:
            raise ValueError(f"{path}: episode_index mismatch at row {index}")
        if record.get("kg_serialization") != expected_serialization:
            raise ValueError(
                f"{path}: kg_serialization={record.get('kg_serialization')!r}, "
                f"expected {expected_serialization!r}"
            )
        if float(record.get("success_threshold_m", -1)) != success_threshold:
            raise ValueError(f"{path}: success threshold mismatch at row {index}")
        if record.get("backend") != "deepseek" or record.get("model") != "deepseek-v4-flash":
            raise ValueError(f"{path}: backend/model mismatch at row {index}")
        if record.get("decision_history") != "on" or record.get("semantic_source") != "predicted":
            raise ValueError(f"{path}: history/semantic mismatch at row {index}")
        if record.get("status") != "complete":
            raise ValueError(f"{path}: incomplete row {index}")
        for metric in METRICS:
            value = record.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: invalid {metric} at row {index}: {value!r}")
        if not 0 <= float(record["habitat_success"]) <= 1:
            raise ValueError(f"{path}: success outside [0,1] at row {index}")
        if not 0 <= float(record["habitat_spl"]) <= 1:
            raise ValueError(f"{path}: SPL outside [0,1] at row {index}")
        if float(record["distance_to_goal"]) < 0 or int(record["steps"]) < 0:
            raise ValueError(f"{path}: negative DTG/steps at row {index}")
        scene = str(record.get("scene", ""))
        episode_id = str(record.get("episode_id", ""))
        if not scene or not episode_id:
            raise ValueError(f"{path}: missing compound episode key at row {index}")
        manifest.append({
            "episode_index": index,
            "episode_id": episode_id,
            "scene": scene,
            "goal": record.get("goal"),
            "episode_key": f"{scene}::{episode_id}",
        })
    if len({row["episode_key"] for row in manifest}) != expected_n:
        raise ValueError(f"{path}: duplicate compound episode key")
    return manifest


def annotate(records: list[dict], manifest: list[dict], split: str) -> list[dict]:
    output = []
    for record, episode in zip(records, manifest):
        row = dict(record)
        row["_split"] = split
        row["_episode_key"] = episode["episode_key"]
        output.append(row)
    return output


def stratified_bootstrap_mean(
    records: list[dict], metric: str, rng: np.random.Generator, draws: int,
) -> list[float]:
    total = len(records)
    samples = np.zeros(draws, dtype=np.float64)
    for split in SPLITS:
        values = np.asarray(
            [float(row[metric]) for row in records if row["_split"] == split],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        samples += values[indices].sum(axis=1) / total
    return percentile_ci(samples)


def stratified_paired_delta(
    left: list[dict],
    right: list[dict],
    metric: str,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, list[float]]:
    total = len(left)
    all_deltas = []
    samples = np.zeros(draws, dtype=np.float64)
    for split in SPLITS:
        left_by_key = {
            row["_episode_key"]: row for row in left if row["_split"] == split
        }
        right_by_key = {
            row["_episode_key"]: row for row in right if row["_split"] == split
        }
        if set(left_by_key) != set(right_by_key):
            raise ValueError(f"paired keys differ for split={split}")
        keys = sorted(left_by_key)
        deltas = np.asarray([
            float(left_by_key[key][metric]) - float(right_by_key[key][metric])
            for key in keys
        ], dtype=np.float64)
        all_deltas.extend(deltas.tolist())
        indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
        samples += deltas[indices].sum(axis=1) / total
    return float(np.mean(all_deltas)), percentile_ci(samples)


def metric_summary(
    label: str,
    records: list[dict],
    rng: np.random.Generator,
    draws: int,
) -> dict:
    output = {
        "kg_format": label,
        "n": len(records),
        "val_mini_n": sum(row["_split"] == "val_mini" for row in records),
        "val_60_n": sum(row["_split"] == "val_60" for row in records),
        "successes": int(sum(int(row["habitat_success"]) for row in records)),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in records]
        ci = stratified_bootstrap_mean(records, metric, rng, draws)
        output[metric] = float(np.mean(values))
        output[f"{metric}_ci95"] = ci
        output[f"{metric}_ci95_low"] = ci[0]
        output[f"{metric}_ci95_high"] = ci[1]
    return output


def split_summary(grouped: dict[str, list[dict]]) -> list[dict]:
    output = []
    for label in FORMATS:
        for split in SPLITS:
            records = [row for row in grouped[label] if row["_split"] == split]
            row = {
                "kg_format": label,
                "split": split,
                "n": len(records),
                "successes": int(sum(int(record["habitat_success"]) for record in records)),
            }
            for metric in METRICS:
                row[metric] = float(np.mean([record[metric] for record in records]))
            output.append(row)
    return output


def category_summary(grouped: dict[str, list[dict]]) -> list[dict]:
    output = []
    for label in FORMATS:
        by_goal = defaultdict(list)
        for record in grouped[label]:
            by_goal[record["goal"]].append(record)
        for goal, records in sorted(by_goal.items()):
            output.append({
                "kg_format": label,
                "goal": goal,
                "n": len(records),
                "habitat_success": float(np.mean([row["habitat_success"] for row in records])),
                "habitat_spl": float(np.mean([row["habitat_spl"] for row in records])),
                "distance_to_goal": float(np.mean([row["distance_to_goal"] for row in records])),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    for split_arg in ("valmini", "val60"):
        for label in FORMATS:
            parser.add_argument(f"--{split_arg}-{label}", required=True)
    parser.add_argument("--out-dir", default="results/aggregated_kg_formats_90")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_719)
    parser.add_argument("--success-threshold", type=float, default=0.2)
    args = parser.parse_args()

    paths = {
        "val_mini": {
            label: Path(getattr(args, f"valmini_{label}")) for label in FORMATS
        },
        "val_60": {
            label: Path(getattr(args, f"val60_{label}")) for label in FORMATS
        },
    }
    for split_paths in paths.values():
        for path in split_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)

    records_by_split: dict[str, dict[str, list[dict]]] = {
        split: {} for split in SPLITS
    }
    manifests: dict[str, dict[str, list[dict]]] = {split: {} for split in SPLITS}

    for label in FORMATS:
        mini_path = paths["val_mini"][label]
        mini_records = load_jsonl(mini_path)
        mini_manifest = load_episode_manifest(mini_path.with_suffix(".log"), 30)
        audit_legacy_run(
            label, mini_path, mini_records, mini_manifest, 30,
            args.success_threshold,
        )
        records_by_split["val_mini"][label] = mini_records
        manifests["val_mini"][label] = mini_manifest

        val60_path = paths["val_60"][label]
        val60_records = load_jsonl(val60_path)
        val60_manifest = audit_val60(
            label, val60_path, val60_records, 60, args.success_threshold
        )
        records_by_split["val_60"][label] = val60_records
        manifests["val_60"][label] = val60_manifest

    for split in SPLITS:
        reference = manifests[split]["text"]
        for label in FORMATS:
            if manifests[split][label] != reference:
                raise ValueError(f"episode manifest differs: split={split}, format={label}")
    mini_keys = {row["episode_key"] for row in manifests["val_mini"]["text"]}
    val60_keys = {row["episode_key"] for row in manifests["val_60"]["text"]}
    if mini_keys & val60_keys:
        raise ValueError("val_mini and val_60 episode keys overlap")

    grouped = {}
    for label in FORMATS:
        grouped[label] = (
            annotate(
                records_by_split["val_mini"][label],
                manifests["val_mini"][label],
                "val_mini",
            )
            + annotate(
                records_by_split["val_60"][label],
                manifests["val_60"][label],
                "val_60",
            )
        )
        if len(grouped[label]) != 90:
            raise ValueError(f"format={label}: combined N is not 90")

    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = [
        metric_summary(label, grouped[label], rng, args.bootstrap_draws)
        for label in FORMATS
    ]
    comparisons = []
    for left_label, right_label in (
        ("json", "text"), ("triples", "text"), ("triples", "json")
    ):
        row = {
            "comparison": f"{left_label}_minus_{right_label}",
            "left": left_label,
            "right": right_label,
            "bootstrap": "stratified by split (30 val_mini + 60 val_60)",
        }
        for metric in METRICS:
            mean, ci = stratified_paired_delta(
                grouped[left_label], grouped[right_label], metric,
                rng, args.bootstrap_draws,
            )
            row[f"delta_{metric}"] = mean
            row[f"delta_{metric}_ci95"] = ci
            row[f"delta_{metric}_ci95_low"] = ci[0]
            row[f"delta_{metric}_ci95_high"] = ci[1]
        comparisons.append(row)

    output = {
        "schema_version": 1,
        "experiment": "mindnav_kg_formats_combined_90",
        "dataset_union": ["HM3D ObjectNav val_mini", "HM3D ObjectNav val_60"],
        "records": 270,
        "unique_episodes_per_format": 90,
        "split_weights": {"val_mini": 30, "val_60": 60},
        "success_threshold_m": args.success_threshold,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_method": "split-stratified episode bootstrap",
        "audit": {
            "three_formats_n90": True,
            "within_split_manifests_identical_across_formats": True,
            "valmini_val60_episode_keys_disjoint": True,
            "primary_metrics": ["habitat_success", "habitat_spl", "distance_to_goal"],
            "excluded_legacy_metrics": [
                "mcoconav_success", "mcoconav_spl",
                "mcoconav_any_agent_success", "mcoconav_navigation_success",
            ],
            "controlled_kg_only_ablation": False,
            "comparability_limitation": "The legacy val_mini text run predates the fixed-order code snapshot used by its JSON/triples runs. The pooled N=90 table is descriptive and should not be treated as a serialization-only causal ablation.",
        },
        "combined_runs": summaries,
        "split_runs": split_summary(grouped),
        "paired_comparisons": comparisons,
        "category_diagnostics": category_summary(grouped),
        "mapping_diagnostics": {
            "val_mini": {
                label: mapping_diagnostics(path)
                for label, path in paths["val_mini"].items()
            },
            "note": "Only val_mini mapping diagnostics are included; the val_60 text mapping log is split across resume artifacts.",
        },
        "episode_manifests": {
            split: manifests[split]["text"] for split in SPLITS
        },
        "sources": {
            split: {label: str(path) for label, path in split_paths.items()}
            for split, split_paths in paths.items()
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "combined_run_summary.csv", summaries)
    write_csv(out_dir / "split_run_summary.csv", output["split_runs"])
    write_csv(out_dir / "paired_comparisons.csv", comparisons)
    write_csv(out_dir / "category_summary.csv", output["category_diagnostics"])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
