#!/usr/bin/env python
"""Replay legacy mapping audits through the current-only frontier view."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_SRC = REPO_ROOT / "code" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from kg_construction import CurrentFrontierView


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_jsonl", type=Path)
    return parser.parse_args()


def enriched_from_record(record):
    enriched = []
    for active in record["active_frontiers"]:
        enriched.append({
            "idx": active["frontier_idx"],
            "centroid": active["centroid"],
            "area": active.get("area", 0),
            "room_type": active.get("room_type", "unknown"),
            "room_confidence": active.get("room_confidence", 0.0),
            "target_prior": active.get(
                "python_prior",
                active.get("target_prior", 0.0),
            ),
        })
    return enriched


def main():
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.mapping_jsonl.read_text().splitlines()
        if line.strip()
    ]
    summary = {
        "decisions": len(records),
        "active_frontiers": 0,
        "preserved_frontiers": 0,
        "collision_decisions": 0,
        "legacy_overwritten_frontiers": 0,
        "legacy_noncurrent_llm_assignments": 0,
        "rejected_noncurrent_llm_assignments": 0,
        "simulated_assignments": 0,
        "simulated_executed_noncurrent": 0,
        "simulated_duplicate_frontier_decisions": 0,
        "room_id_reconstruction_mismatches": 0,
    }

    for record in records:
        view = CurrentFrontierView.from_enriched(
            enriched_from_record(record)
        )
        summary["active_frontiers"] += len(view.frontiers)
        summary["preserved_frontiers"] += sum(
            len(indices) for indices in view.room_to_frontiers.values()
        )
        if any(len(indices) > 1 for indices in view.room_to_frontiers.values()):
            summary["collision_decisions"] += 1
        summary["legacy_overwritten_frontiers"] += record.get(
            "counts", {}
        ).get("overwritten_active_frontiers", 0)

        logged_room_by_idx = {
            int(active["frontier_idx"]): active["room_id"]
            for active in record["active_frontiers"]
        }
        summary["room_id_reconstruction_mismatches"] += sum(
            view.frontiers[idx].room_id != room_id
            for idx, room_id in logged_room_by_idx.items()
        )

        raw_assignments = record.get("llm_room_assignments", {})
        used = set()
        simulated = {}
        for robot_id in sorted(raw_assignments):
            room_id = raw_assignments[robot_id]
            if room_id not in view.room_ids:
                summary["legacy_noncurrent_llm_assignments"] += 1
                summary["rejected_noncurrent_llm_assignments"] += 1
                room_id = None

            frontier_idx = None
            if room_id is not None:
                frontier_idx = view.select_frontier(
                    room_id=room_id,
                    excluded=used,
                )
            if frontier_idx is None:
                frontier_idx = view.select_frontier(excluded=used)
            if frontier_idx is None:
                frontier_idx = view.select_frontier()

            simulated[robot_id] = frontier_idx
            used.add(frontier_idx)
            summary["simulated_assignments"] += 1
            if frontier_idx not in view.frontiers:
                summary["simulated_executed_noncurrent"] += 1

        if (len(view.frontiers) >= len(simulated)
                and len(simulated) != len(set(simulated.values()))):
            summary["simulated_duplicate_frontier_decisions"] += 1

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
