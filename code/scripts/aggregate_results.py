"""Audit and aggregate the six MindNav HM3D experiment runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = ("habitat_success", "habitat_spl", "distance_to_goal")
TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cached_input_tokens",
    "reasoning_tokens", "api_calls",
)
PAIRS = {
    "M-Q7B-TEXT_vs_M-DS-TEXT": ("M-Q7B-TEXT", "M-DS-TEXT"),
    "A-KG-JSON_vs_M-DS-TEXT": ("A-KG-JSON", "M-DS-TEXT"),
    "A-KG-TRIPLES_vs_M-DS-TEXT": ("A-KG-TRIPLES", "M-DS-TEXT"),
    "M-DS-TEXT_vs_A-NO-HISTORY": ("M-DS-TEXT", "A-NO-HISTORY"),
    "A-GT-SEM_vs_M-DS-TEXT": ("A-GT-SEM", "M-DS-TEXT"),
}


def episode_key(record):
    """Return the dataset-unique episode key (IDs repeat across scenes)."""
    return f"{record.get('scene', '')}::{record.get('episode_id', '')}"


def percentile_ci(samples: np.ndarray) -> list[float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def bootstrap_mean(values, rng, draws):
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    return percentile_ci(array[indices].mean(axis=1))


def paired_bootstrap_delta(left, right, rng, draws):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    return float(delta.mean()), percentile_ci(delta[indices].mean(axis=1))


def load_records(paths):
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}")
            record["_source"] = str(path)
            records.append(record)
    return records


def audit_run(run_id, records, expected_n, success_threshold):
    errors = []
    if len(records) != expected_n:
        errors.append(f"N={len(records)}, expected {expected_n}")
    episode_ids = [episode_key(record) for record in records]
    duplicates = sorted(
        episode_id for episode_id, count in Counter(episode_ids).items()
        if not episode_id or count != 1
    )
    if duplicates:
        errors.append(f"missing/duplicate episode keys: {duplicates[:10]}")

    episode_indices = [record.get("episode_index") for record in records]
    if episode_indices != list(range(expected_n)):
        errors.append(
            f"episode_index sequence mismatch: {episode_indices[:10]}..."
        )

    for index, record in enumerate(records):
        label = f"record {index} ({record.get('episode_id')})"
        for metric in METRICS:
            value = record.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{label}: invalid {metric}={value!r}")
        dtg = record.get("distance_to_goal", -1)
        if isinstance(dtg, (int, float)) and dtg < 0:
            errors.append(f"{label}: negative distance_to_goal={dtg}")
        for metric in ("habitat_success", "habitat_spl"):
            value = record.get(metric, -1)
            if isinstance(value, (int, float)) and not 0 <= value <= 1:
                errors.append(f"{label}: out-of-range {metric}={value}")
        for field in TOKEN_FIELDS:
            value = record.get(field)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{label}: invalid {field}={value!r}")
        threshold = record.get("success_threshold_m")
        if threshold != success_threshold:
            errors.append(
                f"{label}: success_threshold_m={threshold}, "
                f"expected {success_threshold}"
            )
        if record.get("status") != "complete":
            errors.append(f"{label}: status={record.get('status')!r}")
        if any(key.startswith("mcoconav") for key in record):
            errors.append(f"{label}: prohibited MCoCoNav field present")
    if errors:
        raise ValueError(f"Audit failed for {run_id}:\n- " + "\n- ".join(errors))
    return episode_ids


def summarize_run(run_id, records, rng, draws):
    summary = {"run_id": run_id, "n": len(records)}
    for metric in METRICS:
        values = [record[metric] for record in records]
        summary[metric] = float(np.mean(values))
        summary[f"{metric}_ci95"] = bootstrap_mean(values, rng, draws)
    for field in TOKEN_FIELDS:
        total = int(sum(record[field] for record in records))
        summary[f"{field}_total"] = total
        summary[f"{field}_per_episode"] = total / len(records)
    for field in (
        "invalid_outputs", "repair_calls", "outer_retries",
        "fallback_decisions", "planning_rounds",
    ):
        summary[f"{field}_total"] = int(
            sum(record.get(field, 0) for record in records)
        )
    summary.update({
        "backend": records[0].get("backend"),
        "model": records[0].get("model"),
        "kg_serialization": records[0].get("kg_serialization"),
        "decision_history": records[0].get("decision_history"),
        "semantic_source": records[0].get("semantic_source"),
        "started_at": min(record["started_at"] for record in records),
        "ended_at": max(record["ended_at"] for record in records),
        "duration_seconds": float(sum(
            record.get("duration_seconds", 0.0) for record in records
        )),
    })
    return summary


def category_summary(grouped):
    output = {}
    for run_id, records in grouped.items():
        by_goal = defaultdict(list)
        for record in records:
            by_goal[record["goal"]].append(record["habitat_success"])
        output[run_id] = {
            goal: {"n": len(values), "sr": float(np.mean(values))}
            for goal, values in sorted(by_goal.items())
        }
    return output


def write_csv(path, rows):
    if not rows:
        return
    scalar_keys = []
    for row in rows:
        for key, value in row.items():
            if key not in scalar_keys and not isinstance(value, (dict, list)):
                scalar_keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in scalar_keys}
            for row in rows
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/aggregated")
    parser.add_argument("--expected-n", type=int, default=60)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_718)
    parser.add_argument("--success-threshold", type=float, default=0.2)
    args = parser.parse_args()

    paths = [Path(path) for path in args.logs]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result logs: {missing}")
    grouped = defaultdict(list)
    for record in load_records(paths):
        grouped[str(record.get("run_id") or record.get("method"))].append(record)

    episode_orders = {}
    for run_id, records in grouped.items():
        records.sort(key=lambda record: int(record["episode_index"]))
        episode_orders[run_id] = audit_run(
            run_id, records, args.expected_n, args.success_threshold
        )
    reference_id = next(iter(grouped))
    reference_set = set(episode_orders[reference_id])
    for run_id, order in episode_orders.items():
        if set(order) != reference_set:
            raise ValueError(
                f"Episode ID set differs: {run_id} vs {reference_id}"
            )

    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = [
        summarize_run(run_id, records, rng, args.bootstrap_draws)
        for run_id, records in sorted(grouped.items())
    ]
    paired = []
    for label, (left_id, right_id) in PAIRS.items():
        if left_id not in grouped or right_id not in grouped:
            continue
        left = {episode_key(record): record for record in grouped[left_id]}
        right = {episode_key(record): record for record in grouped[right_id]}
        row = {"comparison": label, "left": left_id, "right": right_id}
        for metric in METRICS:
            ids = sorted(reference_set)
            mean, ci = paired_bootstrap_delta(
                [left[episode_id][metric] for episode_id in ids],
                [right[episode_id][metric] for episode_id in ids],
                rng,
                args.bootstrap_draws,
            )
            row[f"delta_{metric}"] = mean
            row[f"delta_{metric}_ci95"] = ci
        paired.append(row)

    output = {
        "schema_version": 1,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_draws": args.bootstrap_draws,
        "success_threshold_m": args.success_threshold,
        "runs": summaries,
        "paired_comparisons": paired,
        "category_diagnostics": category_summary(grouped),
        "episode_keys": episode_orders[reference_id],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    write_csv(out_dir / "run_summary.csv", summaries)
    write_csv(out_dir / "paired_comparisons.csv", paired)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
