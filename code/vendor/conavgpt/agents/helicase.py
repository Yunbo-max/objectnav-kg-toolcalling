"""
Helicase: Knowledge Graph-Driven Tool Calling for Embodied Navigation.

Like the enzyme that unwinds DNA to read unknown sequences, Helicase
unwinds the unknown environment by maintaining a Knowledge Graph (KG)
and using LLM tool calling to resolve uncertain nodes.

KG Structure:
    Nodes: Area (room/frontier), Object, Robot
    Edges: connected_to, contains, suggests_room, explored_by, adjacent_to
    Each node has: certainty ∈ [0,1], position (x,y)

Brain Loop:
    1. Update KG from map tools
    2. Find uncertain nodes (certainty < threshold)
    3. Brain reasons: which uncertainty matters most for finding target?
    4. Brain selects: which tools resolve that uncertainty?
    5. Execute tools → update KG
    6. Assign robots to most promising uncertain nodes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import json

# Use the new KG with proper object merging + spatial edges
from agents.helicase_kg import KnowledgeGraph, KGNode, KGEdge, KGUpdater


# ═══════════════════════════════════════════════════════
# LEGACY KNOWLEDGE GRAPH (kept for reference, new one in helicase_kg.py)
# ═══════════════════════════════════════════════════════

@dataclass
class KGNode:
    id: str
    node_type: str          # "area", "object", "robot"
    name: str               # "kitchen", "sink", "robot_0"
    certainty: float        # 0.0 = unknown, 1.0 = certain
    position: Tuple[float, float] = (0, 0)
    properties: Dict = field(default_factory=dict)
    # For areas: {"explored": bool, "frontier_idx": int}
    # For objects: {"category": str}

@dataclass
class KGEdge:
    source: str             # node id
    target: str             # node id
    relation: str           # "connected_to", "contains", "suggests_room", "explored_by"
    certainty: float = 1.0
    distance: float = 0.0   # distance in map pixels (for connected_to edges)


class KnowledgeGraph:
    """Navigation Knowledge Graph — the agent's world model."""

    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self._edge_set: Set[Tuple[str, str, str]] = set()

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self._edge_set.clear()

    def add_node(self, node: KGNode):
        if node.id in self.nodes:
            # Update existing — increase certainty, update position
            existing = self.nodes[node.id]
            existing.certainty = max(existing.certainty, node.certainty)
            if node.position != (0, 0):
                existing.position = node.position
            existing.properties.update(node.properties)
        else:
            self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge):
        key = (edge.source, edge.target, edge.relation)
        if key not in self._edge_set:
            self.edges.append(edge)
            self._edge_set.add(key)

    def get_uncertain_nodes(self, threshold=0.5) -> List[KGNode]:
        """Find nodes with certainty below threshold."""
        uncertain = [n for n in self.nodes.values() if n.certainty < threshold]
        # Sort by certainty ascending (most uncertain first)
        uncertain.sort(key=lambda n: n.certainty)
        return uncertain

    def get_area_nodes(self) -> List[KGNode]:
        return [n for n in self.nodes.values() if n.node_type == "area"]

    def get_object_nodes(self) -> List[KGNode]:
        return [n for n in self.nodes.values() if n.node_type == "object"]

    def get_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """Get (neighbor_id, relation) pairs for a node."""
        neighbors = []
        for e in self.edges:
            if e.source == node_id:
                neighbors.append((e.target, e.relation))
            elif e.target == node_id:
                neighbors.append((e.source, e.relation))
        return neighbors

    def get_objects_in_area(self, area_id: str) -> List[KGNode]:
        """Get all objects contained in an area."""
        objects = []
        for e in self.edges:
            if e.source == area_id and e.relation == "contains":
                if e.target in self.nodes:
                    objects.append(self.nodes[e.target])
        return objects

    def to_text(self, max_nodes=12) -> str:
        """Convert KG to graph text — nodes with edges inline."""
        lines = []

        # Robots first — with path distances to unexplored areas
        robots = [n for n in self.nodes.values() if n.node_type == "robot"]
        for r in robots:
            in_area = None
            paths = []
            explored_areas = []
            for e in self.edges:
                if e.source == r.id:
                    if e.relation == "in":
                        in_area = e.target
                    elif e.relation == "path_to":
                        target_node = self.nodes.get(e.target)
                        if target_node and not target_node.properties.get("explored"):
                            paths.append((e.target, e.distance))
                    elif e.relation == "explored":
                        explored_areas.append(e.target)
            paths.sort(key=lambda x: x[1])
            path_str = ", ".join([f"{pid}({pd:.0f}px)" for pid, pd in paths[:5]])
            lines.append(f"{r.id}: in={in_area}, explored=[{','.join(explored_areas[:4])}]")
            if paths:
                lines.append(f"  paths_to_unexplored: {path_str}")

        lines.append("")

        # Areas — unexplored first, with connections
        areas = self.get_area_nodes()
        unexplored = [a for a in areas if not a.properties.get("explored")]
        explored = [a for a in areas if a.properties.get("explored")]

        if unexplored:
            lines.append(f"UNEXPLORED AREAS ({len(unexplored)}):")
            for a in unexplored[:max_nodes]:
                objs = self.get_objects_in_area(a.id)
                obj_names = [o.name for o in objs]
                # Get connected areas with distances
                connections = []
                for e in self.edges:
                    if e.relation == "connected_to":
                        if e.source == a.id:
                            connections.append(f"{e.target}({e.distance:.0f}px)")
                        elif e.target == a.id:
                            connections.append(f"{e.source}({e.distance:.0f}px)")
                # Get wall separations
                walls = []
                for e in self.edges:
                    if e.relation == "separated_by_wall":
                        if e.source == a.id:
                            walls.append(e.target)
                        elif e.target == a.id:
                            walls.append(e.source)

                finfo = f", frontier_idx={a.properties.get('frontier_idx','?')}" if 'frontier_idx' in a.properties else ""
                line = f"  {a.id}: type={a.name}, certainty={a.certainty:.1f}{finfo}"
                if obj_names:
                    line += f", objects=[{','.join(obj_names)}]"
                if connections:
                    line += f"\n    ──connected_to──> {', '.join(connections[:4])}"
                if walls:
                    line += f"\n    ──wall──> {', '.join(walls[:3])}"
                lines.append(line)

        if explored:
            lines.append(f"\nEXPLORED AREAS ({len(explored)}):")
            for a in explored[:6]:
                objs = self.get_objects_in_area(a.id)
                obj_names = [o.name for o in objs]
                lines.append(f"  {a.id}: type={a.name}, objects=[{','.join(obj_names)}]")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# KG UPDATER — builds KG from map tools
# ═══════════════════════════════════════════════════════

class KGUpdater:
    """Updates KG from semantic map and tool results."""

    ROOM_HINTS = {
        "toilet": "bathroom", "sink": "bathroom", "bathtub": "bathroom",
        "shower": "bathroom", "towel": "bathroom",
        "bed": "bedroom", "chest_of_drawers": "bedroom",
        "sofa": "living_room", "tv_monitor": "living_room",
        "fireplace": "living_room",
        "table": "kitchen", "chair": "living_room",
        "plant": "living_room",
    }

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._explored_areas = set()
        self._area_grid_size = 50  # pixels — areas within this distance merge

    def _pos_to_area_id(self, y, x):
        """Position-based area ID — stable across steps."""
        gy = int(y // self._area_grid_size)
        gx = int(x // self._area_grid_size)
        return f"area_{gy}_{gx}"

    def _infer_room_type(self, obj_names):
        """Infer room type from nearby objects."""
        obj_set = set(obj_names)
        if obj_set & {"toilet", "bathtub", "shower", "sink", "towel"}:
            return "likely_bathroom", 0.5
        elif obj_set & {"bed", "chest_of_drawers"}:
            return "likely_bedroom", 0.5
        elif obj_set & {"sofa", "tv_monitor", "fireplace"}:
            return "likely_living_room", 0.5
        elif obj_set & {"table", "appliances"}:
            return "likely_kitchen", 0.5
        return "unknown", 0.1

    def update_from_map(self, enriched_frontiers, object_list, pose_pred,
                        wall_list, full_map_pred, target_name):
        """Update KG from current map state."""

        # ── Update robot positions ──
        for i, pos in enumerate(pose_pred):
            robot_id = f"robot_{i}"
            self.kg.add_node(KGNode(
                id=robot_id, node_type="robot", name=robot_id,
                certainty=1.0, position=(pos[0], pos[1])
            ))
            # Mark robot's area as explored
            robot_area_id = self._pos_to_area_id(pos[0], pos[1])
            self.kg.add_node(KGNode(
                id=robot_area_id, node_type="area", name="explored",
                certainty=0.8, position=(pos[0], pos[1]),
                properties={"explored": True}
            ))
            self.kg.add_edge(KGEdge(robot_id, robot_area_id, "in", distance=0))
            self.kg.add_edge(KGEdge(robot_id, robot_area_id, "explored"))
            self._explored_areas.add(robot_area_id)

            # Add path distances from robot to all unexplored areas
            for area in self.kg.get_area_nodes():
                if area.properties.get("explored"):
                    continue
                d = np.sqrt((pos[0] - area.position[0])**2 + (pos[1] - area.position[1])**2)
                if d < 300:
                    self.kg.add_edge(KGEdge(robot_id, area.id, "path_to", distance=d))

        # ── Update areas from frontiers (uncertain = unexplored) ──
        for ef in enriched_frontiers:
            cy, cx = ef['centroid'][0], ef['centroid'][1]
            area_id = self._pos_to_area_id(cy, cx)
            objs_nearby = ef.get('nearby_objects', [])
            area_name, cert = self._infer_room_type(objs_nearby)

            # If area was already explored, increase certainty
            if area_id in self._explored_areas:
                cert = max(cert, 0.8)

            self.kg.add_node(KGNode(
                id=area_id, node_type="area", name=area_name,
                certainty=cert,
                position=(cy, cx),
                properties={"frontier_idx": ef['idx'], "size": ef['area'],
                           "explored": area_id in self._explored_areas}
            ))

            # Add nearby objects as nodes + edges
            for obj_name in objs_nearby:
                obj_id = f"obj_{obj_name}_{area_id}"
                self.kg.add_node(KGNode(
                    id=obj_id, node_type="object", name=obj_name,
                    certainty=0.7, position=(cy, cx),
                    properties={"category": obj_name}
                ))
                self.kg.add_edge(KGEdge(area_id, obj_id, "contains"))

                # Semantic: object suggests room type
                if obj_name in self.ROOM_HINTS:
                    room_type = self.ROOM_HINTS[obj_name]
                    self.kg.add_edge(KGEdge(obj_id, f"roomtype_{room_type}", "suggests_room"))

        # ── Walls → "blocked_by" edges between areas ──
        if wall_list is not None and len(wall_list) > 0:
            for i, wall in enumerate(wall_list):
                try:
                    coords = wall[0]  # (y1, x1, y2, x2)
                    wall_mid_y = (coords[0] + coords[2]) / 2
                    wall_mid_x = (coords[1] + coords[3]) / 2
                    wall_id = f"wall_{i}"
                    self.kg.add_node(KGNode(
                        id=wall_id, node_type="wall", name="wall",
                        certainty=0.9, position=(wall_mid_y, wall_mid_x)
                    ))

                    # Find areas on both sides of wall
                    wall_len = np.sqrt((coords[2]-coords[0])**2 + (coords[3]-coords[1])**2)
                    if wall_len > 10:
                        # Normal to wall direction
                        dy = coords[2] - coords[0]
                        dx = coords[3] - coords[1]
                        norm_y, norm_x = -dx / wall_len * 30, dy / wall_len * 30

                        side_a = self._pos_to_area_id(wall_mid_y + norm_y, wall_mid_x + norm_x)
                        side_b = self._pos_to_area_id(wall_mid_y - norm_y, wall_mid_x - norm_x)

                        if side_a != side_b:
                            self.kg.add_edge(KGEdge(side_a, side_b, "separated_by_wall"))
                except:
                    pass

        # ── Connect nearby areas with distance ──
        areas = self.kg.get_area_nodes()
        for i, a1 in enumerate(areas):
            for j, a2 in enumerate(areas):
                if i >= j:
                    continue
                dist = np.sqrt((a1.position[0] - a2.position[0])**2 +
                              (a1.position[1] - a2.position[1])**2)
                if dist < 150:
                    key = (a1.id, a2.id, "separated_by_wall")
                    if key not in self.kg._edge_set:
                        self.kg.add_edge(KGEdge(a1.id, a2.id, "connected_to", distance=dist))

        # ── Objects from full semantic map (Detectron2) ──
        if object_list:
            for obj_name, positions in object_list.items():
                for idx, pos_data in enumerate(positions[:3]):
                    try:
                        coords = pos_data[0][0] if len(pos_data) > 0 else (0, 0)
                        obj_pos = (float(coords[0]), float(coords[1]))
                    except:
                        obj_pos = (0, 0)

                    if obj_pos == (0, 0):
                        continue

                    obj_area_id = self._pos_to_area_id(obj_pos[0], obj_pos[1])
                    obj_id = f"obj_{obj_name}_{obj_area_id}"

                    self.kg.add_node(KGNode(
                        id=obj_id, node_type="object", name=obj_name,
                        certainty=0.9, position=obj_pos,
                        properties={"category": obj_name}
                    ))
                    self.kg.add_edge(KGEdge(obj_area_id, obj_id, "contains"))

                    if obj_name in self.ROOM_HINTS:
                        self.kg.add_edge(KGEdge(obj_id, f"roomtype_{self.ROOM_HINTS[obj_name]}", "suggests_room"))

        # ── Check if target is on map ──
        if target_name and full_map_pred is not None:
            from constants import hm3d_category
            sem = full_map_pred[4:]
            for i, cat in enumerate(hm3d_category):
                if cat == target_name and i < sem.shape[0]:
                    count = (sem[i] > 0.1).sum().item()
                    if count > 5:
                        import torch
                        ys, xs = torch.where(sem[i] > 0.1)
                        target_pos = (int(ys.float().mean()), int(xs.float().mean()))
                        self.kg.add_node(KGNode(
                            id=f"TARGET_{target_name}",
                            node_type="object", name=target_name,
                            certainty=0.95, position=target_pos,
                            properties={"category": target_name, "is_target": True}
                        ))


# ═══════════════════════════════════════════════════════
# HELICASE BRAIN — reasons over KG to select tools & assign robots
# ═══════════════════════════════════════════════════════

class HelicaseBrain:
    """LLM brain that reasons over KG to decide tool calls and robot assignments."""

    def __init__(self, brain, num_agents=2):
        self.brain = brain  # APIBrain or LocalBrain
        self.num_agents = num_agents
        self.reflexion_memory = []

    def decide(self, kg: KnowledgeGraph, target_name: str,
               enriched_frontiers, pose_pred, step: int, max_steps: int,
               decision_history: list) -> Tuple[Dict, List[str], str]:
        """
        Main decision loop:
        1. Check if target found in KG
        2. Find uncertain nodes
        3. Brain reasons which to resolve
        4. Brain selects tools per uncertain node
        5. Assign robots

        Returns: (goal_frontiers, tools_called, method_name)
        """

        # ── FIX 1: Priority check — is target already on the semantic map? ──
        target_node = kg.nodes.get(f"TARGET_{target_name}")
        if target_node and target_node.certainty > 0.8:
            target_pos = np.array(target_node.position)

            # Find nearest frontier to target
            best_frontier = 0
            min_dist = float('inf')
            for ef in enriched_frontiers:
                d = np.sqrt((target_pos[0] - ef['centroid'][0])**2 +
                           (target_pos[1] - ef['centroid'][1])**2)
                if d < min_dist:
                    min_dist = d
                    best_frontier = ef['idx']

            # Find which robot is closest to target → send that one directly
            robot_dists = []
            for i, p in enumerate(pose_pred):
                d = np.sqrt((p[0] - target_pos[0])**2 + (p[1] - target_pos[1])**2)
                robot_dists.append((i, d))
            robot_dists.sort(key=lambda x: x[1])

            closest_robot = robot_dists[0][0]
            farthest_robot = robot_dists[-1][0]

            # Closest robot → target, farthest robot → explore (in case detection is wrong)
            goal = {}
            goal[f"robot_{closest_robot}"] = best_frontier
            # Other robot → farthest frontier for backup exploration
            if len(enriched_frontiers) >= 2:
                farthest_frontier = max(enriched_frontiers,
                    key=lambda ef: np.sqrt((ef['centroid'][0] - target_pos[0])**2 +
                                          (ef['centroid'][1] - target_pos[1])**2))['idx']
                goal[f"robot_{farthest_robot}"] = farthest_frontier
            else:
                goal[f"robot_{farthest_robot}"] = best_frontier

            return goal, ["check_target"], "helicase_target_found"

        # Get KG state as text
        kg_text = kg.to_text()

        # Get uncertain nodes
        uncertain = kg.get_uncertain_nodes(0.5)
        uncertain_text = "\n".join([
            f"  {u.id}: type={u.name}, certainty={u.certainty:.1f}, pos=({u.position[0]:.0f},{u.position[1]:.0f})"
            for u in uncertain[:8]
        ])

        # History
        steps_left = max_steps - step
        history_text = ""
        if decision_history:
            history_text = "PAST DECISIONS:\n" + "\n".join([
                f"  Step {d['step']}: robot_0→area_f{d['r0']}, robot_1→area_f{d['r1']}. Found: [{d.get('objects_found','')}]"
                for d in decision_history[-5:]
            ])

        # Memory
        memory_text = ""
        if self.reflexion_memory:
            memory_text = "LESSONS: " + "; ".join(self.reflexion_memory[-3:])

        # ── BRAIN: Estimate P(target|room) from KG, then assign robots ──
        # Get unexplored rooms with frontier indices
        unexplored = []
        for n in kg.get_nodes_by_type("room"):
            if not n.properties.get("explored") and 'frontier_idx' in n.properties:
                unexplored.append(n)

        if not unexplored:
            return {}, ["no_unexplored"], "helicase_no_rooms"

        unexplored_ids = [n.id for n in unexplored]

        # Single brain call: read KG → estimate probabilities → assign robots
        resp = self.brain.call(
            f"TASK: Find '{target_name}' using {self.num_agents} robots.\n"
            f"Step {step}/{max_steps}. Steps left: {steps_left}.\n\n"
            f"KNOWLEDGE GRAPH:\n{kg_text}\n\n"
            f"{history_text}\n"
            f"{memory_text}\n\n"
            f"For each unexplored room, estimate P({target_name} is there) from 0.0 to 1.0.\n"
            f"Use the objects and spatial edges to reason:\n"
            f"- What objects are in/near this room? What room type do they suggest?\n"
            f"- What rooms are connected to it? (bedroom next to bathroom)\n"
            f"- object ──next_to──> object means they're in same area\n\n"
            f"Then assign robots:\n"
            f"- robot_0 → highest P room\n"
            f"- robot_1 → distant room with second highest P (maximize coverage)\n\n"
            f"Output format:\n"
            f"P({target_name}|<room_id>) = <0.0-1.0>\n"
            f"...\n"
            f"robot_0: <room_id>\n"
            f"robot_1: <room_id>",
            max_tokens=300
        )

        # Parse response — extract probabilities and robot assignments
        import re
        goal_frontiers = {}

        # Build mapping: room_id → frontier_idx
        room_to_frontier = {}
        for n in kg.get_nodes_by_type("room"):
            if 'frontier_idx' in n.properties:
                room_to_frontier[n.id] = n.properties['frontier_idx']

        # Extract P(target|room) estimates from brain response
        probs = {}
        for match in re.finditer(r'P\([^|]+\|(room_\d+_\d+)\)\s*=\s*([\d.]+)', resp):
            room_id = match.group(1)
            prob = float(match.group(2))
            probs[room_id] = prob

        # Extract robot assignments — try room_Y_X format
        for match in re.finditer(r'robot_(\d+)\s*:\s*(room_\d+_\d+)', resp):
            robot_id = f"robot_{match.group(1)}"
            room_id = match.group(2)
            if room_id in room_to_frontier:
                goal_frontiers[robot_id] = room_to_frontier[room_id]

        # Fallback: try area_Y_X format
        if "robot_0" not in goal_frontiers:
            for match in re.finditer(r'robot_(\d+)\s*:\s*(area_\d+_\d+)', resp):
                room_id = match.group(2).replace("area_", "room_")
                if room_id in room_to_frontier:
                    goal_frontiers[f"robot_{match.group(1)}"] = room_to_frontier[room_id]

        # Fallback: use brain's probability estimates to pick best rooms
        if "robot_0" not in goal_frontiers and probs:
            sorted_rooms = sorted(probs.items(), key=lambda x: -x[1])
            if sorted_rooms:
                best_room = sorted_rooms[0][0]
                if best_room in room_to_frontier:
                    goal_frontiers["robot_0"] = room_to_frontier[best_room]
                if len(sorted_rooms) > 1:
                    # Pick farthest high-prob room for robot_1
                    best_pos = kg.nodes[best_room].position if best_room in kg.nodes else (0,0)
                    best_dist = 0
                    for room_id, prob in sorted_rooms[1:]:
                        if room_id in kg.nodes and room_id in room_to_frontier:
                            pos = kg.nodes[room_id].position
                            d = np.sqrt((pos[0]-best_pos[0])**2 + (pos[1]-best_pos[1])**2)
                            if d > best_dist:
                                best_dist = d
                                goal_frontiers["robot_1"] = room_to_frontier[room_id]

        # Auto-assign robot_1 if missing
        if "robot_0" in goal_frontiers and "robot_1" not in goal_frontiers:
            r0_idx = goal_frontiers["robot_0"]
            r0_pos = np.array(enriched_frontiers[min(r0_idx, len(enriched_frontiers)-1)]['centroid'])
            best_dist = 0
            best_idx = enriched_frontiers[0]['idx'] if enriched_frontiers else 0
            for ef in enriched_frontiers:
                d = np.linalg.norm(r0_pos - np.array(ef['centroid']))
                if d > best_dist and ef['idx'] != r0_idx:
                    best_dist = d
                    best_idx = ef['idx']
            goal_frontiers["robot_1"] = best_idx

        # Final fallback: brain completely failed
        if "robot_0" not in goal_frontiers and enriched_frontiers:
            goal_frontiers["robot_0"] = enriched_frontiers[0]['idx']
            if len(enriched_frontiers) > 1:
                p0 = np.array(enriched_frontiers[0]['centroid'])
                best = max(enriched_frontiers[1:],
                          key=lambda ef: np.linalg.norm(np.array(ef['centroid']) - p0))
                goal_frontiers["robot_1"] = best['idx']
            else:
                goal_frontiers["robot_1"] = enriched_frontiers[0]['idx']

        # Log brain's probability estimates
        prob_str = ", ".join([f"{r}={p:.2f}" for r,p in sorted(probs.items(), key=lambda x:-x[1])[:4]]) if probs else "none"
        tools_called = ["kg_reason", "kg_assign"]
        return goal_frontiers, tools_called, f"helicase(P={prob_str})"

    def reflect(self, brain, target_name, success, steps, dtg):
        """Post-episode reflection."""
        outcome = "SUCCESS" if success else "FAILED"
        lesson = brain.call(
            f"Episode: {outcome}. Target: {target_name}. Steps: {steps}. Distance: {dtg:.1f}m.\n"
            f"What one lesson to remember?",
            max_tokens=50
        )
        self.reflexion_memory.append(f"[{target_name},{outcome}] {lesson[:80]}")
        if len(self.reflexion_memory) > 20:
            self.reflexion_memory = self.reflexion_memory[-20:]
