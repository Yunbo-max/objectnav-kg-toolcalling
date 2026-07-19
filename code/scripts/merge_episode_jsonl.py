#!/usr/bin/env python3
"""Merge resumable per-episode JSONL shards with strict ID auditing."""

import argparse
import json
from pathlib import Path


DIAGNOSTIC_FIELDS = (
    "invalid_outputs",
    "repair_calls",
    "outer_retries",
    "fallback_decisions",
    "planning_rounds",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    args = parser.parse_args()

    records = []
    for input_name in args.inputs:
        with Path(input_name).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source"] = f"{input_name}:{line_number}"
                records.append(record)

    by_index = {}
    for record in records:
        index = int(record["episode_index"])
        if index in by_index:
            raise ValueError(
                f"duplicate episode_index={index}: "
                f"{by_index[index]['_source']} and {record['_source']}"
            )
        by_index[index] = record

    expected_indices = set(range(args.expected_n))
    actual_indices = set(by_index)
    if actual_indices != expected_indices:
        raise ValueError(
            f"episode index mismatch: missing="
            f"{sorted(expected_indices - actual_indices)}, extra="
            f"{sorted(actual_indices - expected_indices)}"
        )

    episode_keys = [
        (str(by_index[index]["scene"]),
         str(by_index[index]["episode_id"]))
        for index in range(args.expected_n)
    ]
    if len(set(episode_keys)) != args.expected_n:
        raise ValueError("(scene, episode_id) values are not unique")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for index in range(args.expected_n):
            record = by_index[index]
            record.pop("_source", None)
            for field in DIAGNOSTIC_FIELDS:
                record[field] = int(record.get(field, 0))
            json.dump(record, destination, ensure_ascii=False, sort_keys=True)
            destination.write("\n")

    print(
        f"merged {args.expected_n} unique episodes to {output}; "
        f"first_key={episode_keys[0]} last_key={episode_keys[-1]}"
    )


if __name__ == "__main__":
    main()
