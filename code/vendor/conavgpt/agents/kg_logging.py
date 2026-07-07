"""KG trace export and visualization utilities for MindNav runs."""

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _json_default(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_position(position):
    if position is None:
        return [0.0, 0.0]
    if len(position) < 2:
        return [0.0, 0.0]
    return [_safe_float(position[0]), _safe_float(position[1])]


def serialize_kg(kg) -> Dict:
    nodes = []
    for node in kg.nodes.values():
        nodes.append({
            "id": node.id,
            "type": node.node_type,
            "name": node.name,
            "certainty": _safe_float(node.certainty),
            "position": _safe_position(node.position),
            "properties": dict(node.properties),
        })
    edges = []
    for edge in kg.edges:
        edges.append({
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "distance": _safe_float(edge.distance),
            "properties": dict(edge.properties),
        })
    return {"nodes": nodes, "edges": edges}


def summarize_kg(kg) -> Dict:
    by_type = {}
    by_relation = {}
    for node in kg.nodes.values():
        by_type[node.node_type] = by_type.get(node.node_type, 0) + 1
    for edge in kg.edges:
        by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
    return {
        "num_nodes": len(kg.nodes),
        "num_edges": len(kg.edges),
        "nodes_by_type": by_type,
        "edges_by_relation": by_relation,
    }


def _frontier_for_json(frontier):
    out = {}
    for key, value in frontier.items():
        if key in {"idx", "area"}:
            out[key] = int(value)
        elif key in {"target_prior", "room_confidence"}:
            out[key] = _safe_float(value)
        elif key == "centroid":
            out[key] = _safe_position(value)
        elif key == "nearby_objects":
            out[key] = list(value)
        elif key == "nearby_object_scores":
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def _poses_for_json(poses):
    serial = []
    for pose in poses or []:
        if len(pose) >= 3:
            serial.append({
                "x": _safe_float(pose[0]),
                "y": _safe_float(pose[1]),
                "theta": _safe_float(pose[2]),
            })
    return serial


def _map_array(full_map_pred):
    if full_map_pred is None:
        return None
    if hasattr(full_map_pred, "detach"):
        arr = full_map_pred.detach().cpu().float().numpy()
    else:
        arr = np.asarray(full_map_pred, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.nanmax(arr, axis=0)
    elif arr.ndim != 2:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return None
    max_v = float(np.max(arr))
    if max_v > 0:
        arr = arr / max_v
    return arr


def _method_probabilities(method_name):
    if not method_name or "P=" not in method_name:
        return {}
    text = method_name.split("P=", 1)[1].strip()
    if text.endswith(")"):
        text = text[:-1]
    probs = {}
    for item in text.split(","):
        if ":" not in item:
            continue
        room_id, prob = item.split(":", 1)
        room_id = room_id.strip()
        if room_id:
            probs[room_id] = _safe_float(prob)
    return probs


class KGTraceLogger:
    def __init__(self, trace_dir: str, run_name: str, make_plots: bool = True):
        self.trace_dir = Path(trace_dir)
        self.run_name = run_name
        self.make_plots = make_plots
        self.steps_path = self.trace_dir / "kg_steps.jsonl"
        self.final_path = self.trace_dir / "kg_final.jsonl"
        self.graph_dir = self.trace_dir / "plots" / "graph"
        self.overlay_dir = self.trace_dir / "plots" / "overlay"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path.write_text("")
        self.final_path.write_text("")

    def log_step(
        self,
        episode_idx: int,
        step_idx: int,
        goal_name: str,
        kg,
        robot_poses,
        frontiers,
        assignments,
        tools,
        method_name: str,
        full_map_pred=None,
    ):
        record = {
            "run": self.run_name,
            "episode": int(episode_idx),
            "step": int(step_idx),
            "goal": goal_name,
            "robot_poses": _poses_for_json(robot_poses),
            "frontiers": [_frontier_for_json(f) for f in frontiers],
            "assignments": {k: int(v) for k, v in dict(assignments).items()},
            "tools": list(tools),
            "method": method_name,
            "room_probabilities": _method_probabilities(method_name),
            "kg_summary": summarize_kg(kg),
            "kg": serialize_kg(kg),
        }
        with self.steps_path.open("a") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")
        if self.make_plots:
            self.plot_step(record, full_map_pred=full_map_pred)

    def log_final(self, episode_idx: int, goal_name: str, kg, metrics, steps: int):
        record = {
            "run": self.run_name,
            "episode": int(episode_idx),
            "goal": goal_name,
            "steps": int(steps),
            "metrics": dict(metrics),
            "kg_summary": summarize_kg(kg),
            "kg": serialize_kg(kg),
        }
        with self.final_path.open("a") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")

    def plot_step(self, record: Dict, full_map_pred=None):
        self._plot_graph(record)
        self._plot_overlay(record, full_map_pred)

    def _plot_graph(self, record: Dict):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except Exception as exc:
            self._write_plot_error("graph", record, exc)
            return

        kg = record["kg"]
        graph = nx.MultiDiGraph()
        for node in kg["nodes"]:
            graph.add_node(node["id"], **node)
        for edge in kg["edges"]:
            if edge["source"] in graph and edge["target"] in graph:
                graph.add_edge(edge["source"], edge["target"], **edge)

        if not graph.nodes:
            return

        colors = {
            "room": "#4C78A8",
            "object": "#F58518",
            "robot": "#54A24B",
            "door": "#B279A2",
        }
        node_colors = [colors.get(graph.nodes[n].get("type"), "#9D9DA1") for n in graph.nodes]
        certainties = [graph.nodes[n].get("certainty", 0.5) for n in graph.nodes]
        node_sizes = [250 + 650 * _safe_float(c, 0.5) for c in certainties]
        labels = {
            n: f"{graph.nodes[n].get('name', n)}\n{n}"
            for n in graph.nodes
        }

        pos = nx.spring_layout(graph, seed=record["episode"] * 1000 + record["step"], k=0.7)
        fig, ax = plt.subplots(figsize=(12, 9))
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", alpha=0.35, width=1.2, ax=ax)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)

        edge_labels = {}
        for source, target, key, data in graph.edges(keys=True, data=True):
            if len(edge_labels) >= 24:
                break
            edge_labels[(source, target, key)] = data.get("relation", "")
        try:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)
        except Exception:
            pass

        title = (
            f"{record['run']} ep={record['episode']} step={record['step']} "
            f"goal={record['goal']} nodes={len(kg['nodes'])} edges={len(kg['edges'])}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.graph_dir / self._plot_name(record, "graph"), dpi=160)
        plt.close(fig)

    def _plot_overlay(self, record: Dict, full_map_pred=None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self._write_plot_error("overlay", record, exc)
            return

        arr = _map_array(full_map_pred)
        if arr is None:
            arr = np.zeros((480, 480), dtype=np.float32)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(arr, cmap="gray", origin="upper")

        visible_rooms = {
            node["id"]
            for node in record["kg"]["nodes"]
            if node.get("type") == "room"
            and (
                node.get("properties", {}).get("active_frontier")
                or node.get("properties", {}).get("explored")
            )
        }
        visible_object_ids = set()
        for edge in record["kg"]["edges"]:
            if edge.get("relation") == "contains" and edge.get("source") in visible_rooms:
                visible_object_ids.add(edge.get("target"))

        for frontier in record["frontiers"]:
            y, x = frontier.get("centroid", [0, 0])
            idx = frontier.get("idx", "?")
            prior = frontier.get("target_prior", 0.0)
            ax.scatter([x], [y], marker="x", c="#FFCC00", s=70, linewidths=2)
            ax.text(x + 4, y + 4, f"f{idx} p={prior:.2f}", color="#FFCC00", fontsize=7)

        for node in record["kg"]["nodes"]:
            y, x = node.get("position", [0, 0])
            if y == 0 and x == 0:
                continue
            if node.get("type") == "room":
                if node.get("id") not in visible_rooms:
                    continue
                color = "#4C78A8"
                marker = "o"
            elif node.get("type") == "object":
                if not node.get("properties", {}).get("is_target") and node.get("id") not in visible_object_ids:
                    continue
                color = "#F58518"
                marker = "."
            elif node.get("type") == "robot":
                color = "#54A24B"
                marker = "^"
            else:
                color = "#9D9DA1"
                marker = "."
            ax.scatter([x], [y], c=color, marker=marker, s=45, alpha=0.85)

        for i, pose in enumerate(record["robot_poses"]):
            x = pose.get("x", 0.0)
            y = pose.get("y", 0.0)
            ax.scatter([x], [y], c="#00FF66", marker="^", s=100, edgecolors="black")
            ax.text(x + 5, y + 5, f"r{i}", color="#00FF66", fontsize=9, weight="bold")
            key = f"robot_{i}"
            if key in record["assignments"]:
                target_idx = record["assignments"][key]
                target = next((f for f in record["frontiers"] if f.get("idx") == target_idx), None)
                if target:
                    fy, fx = target.get("centroid", [0, 0])
                    ax.annotate("", xy=(fx, fy), xytext=(x, y),
                                arrowprops={"arrowstyle": "->", "color": "#00FF66", "lw": 1.5})

        ax.set_title(
            f"{record['run']} ep={record['episode']} step={record['step']} goal={record['goal']}",
            fontsize=10,
        )
        ax.set_xlim(0, arr.shape[1])
        ax.set_ylim(arr.shape[0], 0)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(self.overlay_dir / self._plot_name(record, "overlay"), dpi=160)
        plt.close(fig)

    def _plot_name(self, record: Dict, kind: str) -> str:
        return f"ep_{record['episode']:03d}_step_{record['step']:04d}_{kind}.png"

    def _write_plot_error(self, kind: str, record: Dict, exc: Exception):
        err_path = self.trace_dir / f"{kind}_plot_errors.log"
        with err_path.open("a") as fh:
            fh.write(
                f"episode={record.get('episode')} step={record.get('step')} "
                f"error={type(exc).__name__}: {exc}\n"
            )
