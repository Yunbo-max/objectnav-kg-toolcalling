"""Aggregate per-episode logs for Table 1 of the paper."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument("--logs", default=str(repo_root / "results" / "runs"))
    p.add_argument("--out", default=str(repo_root / "results" / "table1.csv"))
    args = p.parse_args()

    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "success": 0.0, "spl": 0.0}
    )
    for f in Path(args.logs).rglob("*.jsonl"):
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            method = rec["method"]
            agg[method]["n"] += 1
            agg[method]["success"] += rec.get("success", 0)
            agg[method]["spl"] += rec.get("spl", 0.0)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "SR", "SPL"])
        for method, d in sorted(agg.items()):
            n = max(d["n"], 1)
            w.writerow([method, int(d["n"]),
                        f"{d['success']/n:.3f}", f"{d['spl']/n:.3f}"])


if __name__ == "__main__":
    main()
