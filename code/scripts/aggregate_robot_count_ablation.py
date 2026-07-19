#!/usr/bin/env python3
"""Audit and aggregate the MindNav 1/2/3-robot ablation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_results import (
    METRICS,
    audit_run,
    episode_key,
    load_records,
    paired_bootstrap_delta,
    summarize_run,
    write_csv,
)


EXPECTED_COUNTS = {
    "A-ROBOTS-1": 1,
    "M-DS-TEXT": 2,
    "A-ROBOTS-3": 3,
}
PAIR_SPECS = (
    ("ROBOTS-2_vs_1", "M-DS-TEXT", "A-ROBOTS-1"),
    ("ROBOTS-3_vs_2", "A-ROBOTS-3", "M-DS-TEXT"),
    ("ROBOTS-3_vs_1", "A-ROBOTS-3", "A-ROBOTS-1"),
)
COMMON_FIELDS = (
    "backend", "model", "kg_serialization", "decision_history",
    "semantic_source", "success_threshold_m", "seed", "seed_mode",
)


def audit_robot_count_metadata(run_id, records):
    expected = EXPECTED_COUNTS[run_id]
    for index, record in enumerate(records):
        if expected == 2:
            # The frozen baseline predates the explicit num_agents field.
            if "num_agents" in record and record["num_agents"] != 2:
                raise ValueError(
                    f"{run_id} record {index}: num_agents="
                    f"{record['num_agents']}"
                )
            continue
        if record.get("num_agents") != expected:
            raise ValueError(
                f"{run_id} record {index}: num_agents="
                f"{record.get('num_agents')}, expected {expected}"
            )
        if record.get("robot_count_ablation") is not True:
            raise ValueError(
                f"{run_id} record {index}: missing ablation marker"
            )
        poses = record.get("initial_agent_poses")
        expected_ids = [f"robot_{i}" for i in range(expected)]
        if (not isinstance(poses, list) or len(poses) != expected
                or [pose.get("robot_id") for pose in poses] != expected_ids):
            raise ValueError(
                f"{run_id} record {index}: invalid initial_agent_poses"
            )
        steps = int(record.get("steps", -1))
        if record.get("team_steps") != steps:
            raise ValueError(
                f"{run_id} record {index}: team_steps mismatch"
            )
        if record.get("robot_actions") != expected * steps:
            raise ValueError(
                f"{run_id} record {index}: robot_actions mismatch"
            )


def audit_common_configuration(grouped):
    baseline = grouped["M-DS-TEXT"][0]
    for run_id, records in grouped.items():
        for field in COMMON_FIELDS:
            values = {record.get(field) for record in records}
            if len(values) != 1:
                raise ValueError(f"{run_id}: non-constant {field}: {values}")
            value = next(iter(values))
            if value != baseline.get(field):
                raise ValueError(
                    f"{run_id}: {field}={value!r}, baseline="
                    f"{baseline.get(field)!r}"
                )


def add_costs(summary, records, num_agents):
    team_steps = [int(record.get("team_steps", record["steps"]))
                  for record in records]
    robot_actions = [int(record.get("robot_actions", num_agents * steps))
                     for record, steps in zip(records, team_steps)]
    durations = [float(record.get("duration_seconds", 0.0))
                 for record in records]
    summary.update({
        "num_agents": int(num_agents),
        "team_steps_total": int(sum(team_steps)),
        "team_steps_per_episode": float(np.mean(team_steps)),
        "robot_actions_total": int(sum(robot_actions)),
        "robot_actions_per_episode": float(np.mean(robot_actions)),
        "duration_seconds_total": float(sum(durations)),
        "duration_seconds_per_episode": float(np.mean(durations)),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-1-log", required=True)
    parser.add_argument("--robot-2-log", required=True)
    parser.add_argument("--robot-3-log", required=True)
    parser.add_argument(
        "--out-dir", default="results/aggregated_robot_count"
    )
    parser.add_argument("--expected-n", type=int, default=60)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_718)
    parser.add_argument("--success-threshold", type=float, default=0.2)
    args = parser.parse_args()

    paths = [Path(args.robot_1_log), Path(args.robot_2_log),
             Path(args.robot_3_log)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result logs: {missing}")

    grouped = defaultdict(list)
    for record in load_records(paths):
        grouped[str(record.get("run_id") or record.get("method"))].append(
            record
        )
    if set(grouped) != set(EXPECTED_COUNTS):
        raise ValueError(
            f"run IDs={sorted(grouped)}, expected={sorted(EXPECTED_COUNTS)}"
        )

    episode_orders = {}
    for run_id, records in grouped.items():
        records.sort(key=lambda record: int(record["episode_index"]))
        episode_orders[run_id] = audit_run(
            run_id, records, args.expected_n, args.success_threshold
        )
        audit_robot_count_metadata(run_id, records)
    audit_common_configuration(grouped)

    reference_order = episode_orders["M-DS-TEXT"]
    reference_set = set(reference_order)
    for run_id, order in episode_orders.items():
        if order != reference_order or set(order) != reference_set:
            raise ValueError(
                f"episode order/key mismatch: {run_id} vs M-DS-TEXT"
            )

    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = []
    for run_id, num_agents in sorted(
        EXPECTED_COUNTS.items(), key=lambda item: item[1]
    ):
        summary = summarize_run(
            run_id, grouped[run_id], rng, args.bootstrap_draws
        )
        add_costs(summary, grouped[run_id], num_agents)
        summaries.append(summary)

    paired = []
    for label, left_id, right_id in PAIR_SPECS:
        left = {episode_key(record): record for record in grouped[left_id]}
        right = {episode_key(record): record for record in grouped[right_id]}
        row = {"comparison": label, "left": left_id, "right": right_id}
        for metric in METRICS:
            mean, ci = paired_bootstrap_delta(
                [left[key][metric] for key in reference_order],
                [right[key][metric] for key in reference_order],
                rng,
                args.bootstrap_draws,
            )
            row[f"delta_{metric}"] = mean
            row[f"delta_{metric}_ci95"] = ci
        paired.append(row)

    output = {
        "schema_version": 1,
        "experiment": "mindnav_robot_count_ablation",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_draws": args.bootstrap_draws,
        "success_threshold_m": args.success_threshold,
        "baseline_reused": "M-DS-TEXT",
        "runs": summaries,
        "paired_comparisons": paired,
        "episode_keys": reference_order,
        "limitation": (
            "M-DS-TEXT was run earlier than A-ROBOTS-1/3; episode bootstrap "
            "does not model remote DeepSeek service-time drift."
        ),
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
