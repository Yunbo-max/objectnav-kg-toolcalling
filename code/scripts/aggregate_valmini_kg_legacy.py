#!/usr/bin/env python3
"""Audit and aggregate the three legacy MindNav val_mini KG runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = ("habitat_success", "habitat_spl", "distance_to_goal", "steps")
PRIMARY_METRICS = METRICS[:3]
EPISODE_START = re.compile(
    r"\[EPISODE_START\] index=(\d+) episode_id=(\S+) "
    r"scene=(\S+) target=(\S+)"
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}")
    return records


def load_episode_manifest(log_path: Path, expected_n: int) -> list[dict]:
    matches = EPISODE_START.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if len(matches) != expected_n:
        raise ValueError(
            f"{log_path}: found {len(matches)} episode starts, expected {expected_n}"
        )
    manifest = [
        {
            "episode_index": int(index),
            "episode_id": episode_id,
            "scene": scene,
            "goal": goal,
            "episode_key": f"{scene}::{episode_id}",
        }
        for index, episode_id, scene, goal in matches
    ]
    if [row["episode_index"] for row in manifest] != list(range(expected_n)):
        raise ValueError(f"{log_path}: episode indices are not 0..{expected_n - 1}")
    if len({row["episode_key"] for row in manifest}) != expected_n:
        raise ValueError(f"{log_path}: duplicate compound episode key")
    return manifest


def audit_run(
    label: str,
    path: Path,
    records: list[dict],
    manifest: list[dict],
    expected_n: int,
    success_threshold: float,
) -> None:
    if len(records) != expected_n:
        raise ValueError(f"{path}: N={len(records)}, expected {expected_n}")
    expected_serialization = {"text": None, "json": "json", "triples": "triples"}[label]
    for index, (record, episode) in enumerate(zip(records, manifest)):
        if int(record.get("episode", -1)) != index + 1:
            raise ValueError(f"{path}: legacy episode sequence mismatch at row {index}")
        if record.get("goal") != episode["goal"]:
            raise ValueError(f"{path}: goal mismatch at row {index}")
        if record.get("kg_serialization") != expected_serialization:
            raise ValueError(
                f"{path}: kg_serialization={record.get('kg_serialization')!r}, "
                f"expected {expected_serialization!r}"
            )
        for metric in METRICS:
            value = record.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: invalid {metric} at row {index}: {value!r}")
        if not 0 <= float(record["habitat_success"]) <= 1:
            raise ValueError(f"{path}: habitat_success outside [0,1] at row {index}")
        if not 0 <= float(record["habitat_spl"]) <= 1:
            raise ValueError(f"{path}: habitat_spl outside [0,1] at row {index}")
        if float(record["distance_to_goal"]) < 0 or int(record["steps"]) < 0:
            raise ValueError(f"{path}: negative DTG/steps at row {index}")
        if float(record.get("success", -1)) != float(record["habitat_success"]):
            raise ValueError(f"{path}: success aliases disagree at row {index}")
        if not math.isclose(
            float(record.get("spl", -1)), float(record["habitat_spl"]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"{path}: SPL aliases disagree at row {index}")
        threshold_success = int(float(record["distance_to_goal"]) < success_threshold)
        if int(record["habitat_success"]) != threshold_success:
            raise ValueError(
                f"{path}: success/DTG threshold mismatch at row {index}"
            )


def percentile_ci(samples: np.ndarray) -> list[float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def bootstrap_mean(values: list[float], rng: np.random.Generator, draws: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    return percentile_ci(array[indices].mean(axis=1))


def paired_delta(
    left: list[float],
    right: list[float],
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, list[float]]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    return float(delta.mean()), percentile_ci(delta[indices].mean(axis=1))


def summarize_run(
    label: str,
    records: list[dict],
    rng: np.random.Generator,
    draws: int,
) -> dict:
    output = {
        "kg_format": label,
        "n": len(records),
        "successes": int(sum(int(row["habitat_success"]) for row in records)),
        "method": records[0].get("method"),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in records]
        output[metric] = float(np.mean(values))
        output[f"{metric}_ci95"] = bootstrap_mean(values, rng, draws)
    return output


def success_contingency(left: list[dict], right: list[dict]) -> dict:
    pairs = [(int(a["habitat_success"]), int(b["habitat_success"])) for a, b in zip(left, right)]
    return {
        "both_success": sum(a == 1 and b == 1 for a, b in pairs),
        "left_only": sum(a == 1 and b == 0 for a, b in pairs),
        "right_only": sum(a == 0 and b == 1 for a, b in pairs),
        "both_fail": sum(a == 0 and b == 0 for a, b in pairs),
    }


def mapping_diagnostics(result_path: Path) -> dict:
    mapping_path = result_path.with_suffix(".mapping.jsonl")
    records = load_jsonl(mapping_path)
    statuses = Counter(
        row.get("decision_audit", {}).get("llm_output_status", "missing")
        for row in records
    )
    fallback_reasons = Counter(
        row.get("decision_audit", {}).get("fallback_reason") or "none"
        for row in records
    )
    tool_sequences = Counter(
        ",".join(str(tool) for tool in row.get("tools", []))
        for row in records
    )
    invalid = int(statuses.get("invalid_after_repair", 0))
    return {
        "source": str(mapping_path),
        "decision_rounds": len(records),
        "llm_output_status": dict(sorted(statuses.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "tool_sequences": dict(sorted(tool_sequences.items())),
        "invalid_after_repair_rate": invalid / len(records),
    }


def category_summary(grouped: dict[str, list[dict]]) -> list[dict]:
    output = []
    for label, records in grouped.items():
        by_goal = defaultdict(list)
        for record in records:
            by_goal[record["goal"]].append(record)
        for goal, rows in sorted(by_goal.items()):
            output.append({
                "kg_format": label,
                "goal": goal,
                "n": len(rows),
                "habitat_success": float(np.mean([row["habitat_success"] for row in rows])),
                "habitat_spl": float(np.mean([row["habitat_spl"] for row in rows])),
                "distance_to_goal": float(np.mean([row["distance_to_goal"] for row in rows])),
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (list, dict)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--triples", required=True)
    parser.add_argument("--out-dir", default="results/aggregated_valmini_kg_90")
    parser.add_argument("--expected-n", type=int, default=30)
    parser.add_argument("--success-threshold", type=float, default=0.2)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_717)
    args = parser.parse_args()

    paths = {label: Path(getattr(args, label)) for label in ("text", "json", "triples")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if not path.with_suffix(".log").is_file():
            raise FileNotFoundError(path.with_suffix(".log"))

    grouped = {label: load_jsonl(path) for label, path in paths.items()}
    manifests = {
        label: load_episode_manifest(path.with_suffix(".log"), args.expected_n)
        for label, path in paths.items()
    }
    for label in grouped:
        audit_run(
            label, paths[label], grouped[label], manifests[label],
            args.expected_n, args.success_threshold,
        )
    reference_manifest = manifests["text"]
    for label, manifest in manifests.items():
        if manifest != reference_manifest:
            raise ValueError(f"episode manifest differs: {label} vs text")

    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = [
        summarize_run(label, grouped[label], rng, args.bootstrap_draws)
        for label in ("text", "json", "triples")
    ]
    comparisons = []
    for left_label, right_label in (
        ("json", "text"), ("triples", "text"), ("triples", "json")
    ):
        left, right = grouped[left_label], grouped[right_label]
        row = {
            "comparison": f"{left_label}_minus_{right_label}",
            "left": left_label,
            "right": right_label,
            "success_contingency": success_contingency(left, right),
        }
        for metric in METRICS:
            mean, ci = paired_delta(
                [float(record[metric]) for record in left],
                [float(record[metric]) for record in right],
                rng, args.bootstrap_draws,
            )
            row[f"delta_{metric}"] = mean
            row[f"delta_{metric}_ci95"] = ci
        comparisons.append(row)

    all_records = [record for records in grouped.values() for record in records]
    pooled = {
        "total_run_records": len(all_records),
        "unique_dataset_episodes": args.expected_n,
        "note": "The same 30 episodes were run once per KG format; these are not 90 independent dataset episodes.",
    }
    for metric in METRICS:
        pooled[metric] = float(np.mean([record[metric] for record in all_records]))

    output = {
        "schema_version": 1,
        "experiment": "legacy_mindnav_valmini_kg_formats",
        "dataset": "HM3D ObjectNav val_mini",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_draws": args.bootstrap_draws,
        "success_threshold_m": args.success_threshold,
        "audit": {
            "records": 3 * args.expected_n,
            "runs": 3,
            "episodes_per_run": args.expected_n,
            "episode_manifests_identical": True,
            "success_matches_dtg_threshold_all_records": True,
            "legacy_success_spl_aliases_match_habitat": True,
            "excluded_metrics": [
                "mcoconav_success",
                "mcoconav_spl",
                "mcoconav_any_agent_success",
                "mcoconav_navigation_success",
            ],
            "exclusion_reason": "Structured-format runs contain zero/discordant MCoCoNav diagnostic fields, so they are not comparable across formats.",
            "controlled_kg_only_ablation": False,
            "comparability_limitation": "The text run predates the fixed-order code snapshot used by the JSON/triples runs; results are descriptive historical comparisons, not a KG-serialization-only causal ablation.",
        },
        "runs": summaries,
        "paired_comparisons": comparisons,
        "pooled_run_record_summary": pooled,
        "category_diagnostics": category_summary(grouped),
        "mapping_diagnostics": {
            label: mapping_diagnostics(path) for label, path in paths.items()
        },
        "episode_manifest": reference_manifest,
        "sources": {label: str(path) for label, path in paths.items()},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(out_dir / "run_summary.csv", summaries)
    write_csv(out_dir / "paired_comparisons.csv", comparisons)
    write_csv(out_dir / "category_summary.csv", output["category_diagnostics"])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
