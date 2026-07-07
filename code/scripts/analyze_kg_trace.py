#!/usr/bin/env python3
"""Analyze MindNav KG traces and decision alignment."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_jsonl(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def safe_mean(values):
    return mean(values) if values else 0.0


def room_id_from_centroid(centroid, grid_scale):
    if not centroid or len(centroid) < 2:
        return None
    return f"room_{int(float(centroid[0]) // grid_scale)}_{int(float(centroid[1]) // grid_scale)}"


def kg_integrity(record, duplicate_radius):
    kg = record.get("kg", {})
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    ids = {node.get("id") for node in nodes}
    node_types = {}
    edge_rels = {}
    for node in nodes:
        node_types[node.get("type")] = node_types.get(node.get("type"), 0) + 1
    for edge in edges:
        edge_rels[edge.get("relation")] = edge_rels.get(edge.get("relation"), 0) + 1

    dangling = sum(
        1 for edge in edges
        if edge.get("source") not in ids or edge.get("target") not in ids
    )

    relations_by_pair = {}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in ids or target not in ids or source == target:
            continue
        pair = tuple(sorted((source, target)))
        relations_by_pair.setdefault(pair, set()).add(edge.get("relation"))
    contradictions = sum(
        1 for rels in relations_by_pair.values()
        if "connected_to" in rels and "separated_by_wall" in rels
    )

    objects = [
        node for node in nodes
        if node.get("type") == "object"
        and not node.get("properties", {}).get("is_target")
        and isinstance(node.get("position"), list)
        and len(node.get("position")) >= 2
    ]
    close_duplicate_pairs = 0
    duplicate_pairs_by_category = {}
    for i, first in enumerate(objects):
        first_category = first.get("properties", {}).get("category") or first.get("name")
        for second in objects[i + 1:]:
            second_category = second.get("properties", {}).get("category") or second.get("name")
            if first_category != second_category:
                continue
            dy = float(first["position"][0]) - float(second["position"][0])
            dx = float(first["position"][1]) - float(second["position"][1])
            if math.hypot(dy, dx) < duplicate_radius:
                close_duplicate_pairs += 1
                duplicate_pairs_by_category[first_category] = duplicate_pairs_by_category.get(first_category, 0) + 1

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "node_types": node_types,
        "edge_relations": edge_rels,
        "door_nodes": node_types.get("door", 0),
        "dangling_edges": dangling,
        "connection_wall_contradictions": contradictions,
        "close_duplicate_pairs": close_duplicate_pairs,
        "duplicate_pairs_by_category": duplicate_pairs_by_category,
    }


def decision_alignment(record, grid_scale):
    frontiers = record.get("frontiers", [])
    assignments = record.get("assignments", {})
    chosen = set(assignments.values())
    out = {
        "episode": record.get("episode"),
        "step": record.get("step"),
        "num_frontiers": len(frontiers),
        "num_chosen": len(chosen),
        "selected_highest_target_prior": None,
        "selected_highest_room_probability": None,
        "chosen_target_prior_mean": None,
        "max_target_prior": None,
        "chosen_room_probability_mean": None,
        "max_room_probability": None,
    }
    if not frontiers or not chosen:
        return out

    priors = {int(f.get("idx")): float(f.get("target_prior", 0.0)) for f in frontiers}
    max_prior = max(priors.values()) if priors else 0.0
    max_prior_idxs = {idx for idx, value in priors.items() if value == max_prior}
    chosen_priors = [priors[idx] for idx in chosen if idx in priors]
    out["max_target_prior"] = max_prior
    out["chosen_target_prior_mean"] = safe_mean(chosen_priors)
    out["selected_highest_target_prior"] = bool(chosen & max_prior_idxs)

    room_probabilities = record.get("room_probabilities", {})
    frontier_probs = {}
    for frontier in frontiers:
        idx = int(frontier.get("idx"))
        room_id = room_id_from_centroid(frontier.get("centroid"), grid_scale)
        if room_id in room_probabilities:
            frontier_probs[idx] = float(room_probabilities[room_id])
    if frontier_probs:
        max_prob = max(frontier_probs.values())
        max_prob_idxs = {idx for idx, value in frontier_probs.items() if value == max_prob}
        chosen_probs = [frontier_probs[idx] for idx in chosen if idx in frontier_probs]
        out["max_room_probability"] = max_prob
        out["chosen_room_probability_mean"] = safe_mean(chosen_probs)
        out["selected_highest_room_probability"] = bool(chosen & max_prob_idxs)

    return out


def summarize(trace_dir, metrics_path=None, duplicate_radius=60.0, grid_scale=40):
    trace_dir = Path(trace_dir)
    final_records = load_jsonl(trace_dir / "kg_final.jsonl")
    step_records = load_jsonl(trace_dir / "kg_steps.jsonl")
    metric_records = load_jsonl(metrics_path) if metrics_path else []

    final_by_episode = {record.get("episode"): record for record in final_records}
    metrics_by_episode = {record.get("episode"): record for record in metric_records}

    episodes = []
    for episode in sorted(final_by_episode):
        final = final_by_episode[episode]
        metrics = dict(final.get("metrics") or metrics_by_episode.get(episode, {}))
        integrity = kg_integrity(final, duplicate_radius)
        episodes.append({
            "episode": episode,
            "goal": final.get("goal"),
            "success": float(metrics.get("success", 0.0)),
            "spl": float(metrics.get("spl", 0.0)),
            "steps": int(final.get("steps", 0)),
            "kg": integrity,
        })

    decisions = [decision_alignment(record, grid_scale) for record in step_records]
    prior_decisions = [d for d in decisions if d["selected_highest_target_prior"] is not None]
    prob_decisions = [d for d in decisions if d["selected_highest_room_probability"] is not None]

    summary = {
        "trace_dir": str(trace_dir),
        "num_episodes": len(episodes),
        "num_decision_steps": len(step_records),
        "sr": safe_mean([episode["success"] for episode in episodes]),
        "spl": safe_mean([episode["spl"] for episode in episodes]),
        "avg_final_nodes": safe_mean([episode["kg"]["nodes"] for episode in episodes]),
        "avg_final_edges": safe_mean([episode["kg"]["edges"] for episode in episodes]),
        "total_door_nodes": sum(episode["kg"]["door_nodes"] for episode in episodes),
        "total_dangling_edges": sum(episode["kg"]["dangling_edges"] for episode in episodes),
        "total_connection_wall_contradictions": sum(
            episode["kg"]["connection_wall_contradictions"] for episode in episodes
        ),
        "total_close_duplicate_pairs": sum(episode["kg"]["close_duplicate_pairs"] for episode in episodes),
        "decision_alignment": {
            "selected_highest_target_prior_rate": safe_mean([
                float(d["selected_highest_target_prior"]) for d in prior_decisions
            ]),
            "selected_highest_room_probability_rate": safe_mean([
                float(d["selected_highest_room_probability"]) for d in prob_decisions
            ]),
            "avg_chosen_target_prior": safe_mean([
                d["chosen_target_prior_mean"] for d in prior_decisions
                if d["chosen_target_prior_mean"] is not None
            ]),
            "avg_max_target_prior": safe_mean([
                d["max_target_prior"] for d in prior_decisions
                if d["max_target_prior"] is not None
            ]),
        },
        "episodes": episodes,
        "decisions": decisions,
    }
    return summary


def write_markdown(summary, path):
    lines = [
        "# KG Trace Analysis",
        "",
        f"- Episodes: {summary['num_episodes']}",
        f"- Decision steps: {summary['num_decision_steps']}",
        f"- SR: {summary['sr']:.4f}",
        f"- SPL: {summary['spl']:.4f}",
        f"- Avg final nodes: {summary['avg_final_nodes']:.1f}",
        f"- Avg final edges: {summary['avg_final_edges']:.1f}",
        f"- Door nodes: {summary['total_door_nodes']}",
        f"- Dangling edges: {summary['total_dangling_edges']}",
        f"- Connected/wall contradictions: {summary['total_connection_wall_contradictions']}",
        f"- Close duplicate pairs: {summary['total_close_duplicate_pairs']}",
        f"- Highest target-prior chosen rate: {summary['decision_alignment']['selected_highest_target_prior_rate']:.4f}",
        f"- Highest room-probability chosen rate: {summary['decision_alignment']['selected_highest_room_probability_rate']:.4f}",
        "",
        "| ep | goal | success | SPL | steps | nodes | edges | dup pairs | dangling | contradictions |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for episode in summary["episodes"]:
        kg = episode["kg"]
        lines.append(
            f"| {episode['episode']} | {episode['goal']} | {episode['success']:.0f} | "
            f"{episode['spl']:.4f} | {episode['steps']} | {kg['nodes']} | {kg['edges']} | "
            f"{kg['close_duplicate_pairs']} | {kg['dangling_edges']} | "
            f"{kg['connection_wall_contradictions']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", help="Directory containing kg_steps.jsonl and kg_final.jsonl")
    parser.add_argument("--metrics", help="Optional per-episode metrics JSONL")
    parser.add_argument("--duplicate-radius", type=float, default=60.0)
    parser.add_argument("--room-grid-scale", type=int, default=40)
    parser.add_argument("--output-dir", help="Defaults to <trace_dir>/analysis")
    args = parser.parse_args()

    summary = summarize(
        args.trace_dir,
        metrics_path=args.metrics,
        duplicate_radius=args.duplicate_radius,
        grid_scale=args.room_grid_scale,
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.trace_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kg_analysis.json"
    md_path = output_dir / "kg_analysis.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_markdown(summary, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"SR={summary['sr']:.4f} SPL={summary['spl']:.4f} "
        f"dup_pairs={summary['total_close_duplicate_pairs']} "
        f"dangling={summary['total_dangling_edges']} "
        f"contradictions={summary['total_connection_wall_contradictions']}"
    )


if __name__ == "__main__":
    main()
