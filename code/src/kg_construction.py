"""
Helicase KG v2 — Proper Knowledge Graph with:
- Object nodes with Detectron2 confidence, merged across detections
- Spatial edges between objects (next_to, near)
- Room certainty from object detection counts
- Connected_via(door) edges between rooms
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


@dataclass
class KGNode:
    id: str
    node_type: str          # "room", "object", "robot", "door"
    name: str               # "bathroom", "sink", "robot_0", "door_1"
    certainty: float        # from detections: 1 - ∏(1-conf_i)
    position: Tuple[float, float] = (0, 0)
    properties: Dict = field(default_factory=dict)
    # For rooms: {"explored": bool, "frontier_idx": int, "observation_count": int}
    # For objects: {"category": str, "detections": [(conf, pos), ...]}


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str       # "contains", "next_to", "near", "connected_to",
                        # "connected_via", "separated_by_wall", "suggests",
                        # "in", "explored", "path_to"
    distance: float = 0.0
    properties: Dict = field(default_factory=dict)


class KnowledgeGraph:
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
            existing = self.nodes[node.id]
            # Merge certainty: 1 - (1-old)(1-new)
            existing.certainty = 1.0 - (1.0 - existing.certainty) * (1.0 - node.certainty)
            existing.certainty = min(existing.certainty, 0.99)
            if node.position != (0, 0):
                # Average position for stability
                if existing.position != (0, 0):
                    existing.position = (
                        0.7 * existing.position[0] + 0.3 * node.position[0],
                        0.7 * existing.position[1] + 0.3 * node.position[1],
                    )
                else:
                    existing.position = node.position
            existing.properties.update(node.properties)
        else:
            self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge):
        key = (edge.source, edge.target, edge.relation)
        if key not in self._edge_set:
            self.edges.append(edge)
            self._edge_set.add(key)
        else:
            # Update distance if edge exists
            for e in self.edges:
                if (e.source, e.target, e.relation) == key:
                    e.distance = edge.distance
                    break

    def get_nodes_by_type(self, node_type: str) -> List[KGNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_uncertain_nodes(self, threshold=0.5) -> List[KGNode]:
        uncertain = [n for n in self.nodes.values() if n.certainty < threshold]
        uncertain.sort(key=lambda n: n.certainty)
        return uncertain

    def get_area_nodes(self) -> List[KGNode]:
        return self.get_nodes_by_type("room")

    def get_edges_from(self, node_id: str, relation: str = None) -> List[KGEdge]:
        edges = []
        for e in self.edges:
            if e.source == node_id or e.target == node_id:
                if relation is None or e.relation == relation:
                    edges.append(e)
        return edges

    def get_objects_in_room(self, room_id: str) -> List[KGNode]:
        objs = []
        for e in self.edges:
            if e.source == room_id and e.relation == "contains":
                if e.target in self.nodes:
                    objs.append(self.nodes[e.target])
        return objs

    def to_text(self) -> str:
        """KG as graph text for LLM — nodes with inline edges."""
        lines = []

        # Robots
        robots = self.get_nodes_by_type("robot")
        for r in robots:
            in_room = None
            paths = []
            explored = []
            for e in self.edges:
                if e.source == r.id:
                    if e.relation == "in":
                        in_room = e.target
                    elif e.relation == "path_to":
                        target = self.nodes.get(e.target)
                        if target and not target.properties.get("explored"):
                            paths.append((e.target, e.distance))
                    elif e.relation == "explored":
                        explored.append(e.target)
            paths.sort(key=lambda x: x[1])
            path_str = ", ".join([f"{p[0]}({p[1]:.0f}px)" for p in paths[:5]])
            lines.append(f"{r.id}: in={in_room}, explored={len(explored)} rooms")
            if paths:
                lines.append(f"  reachable_unexplored: {path_str}")

        lines.append("")

        # Unexplored rooms
        rooms = self.get_nodes_by_type("room")
        unexplored = [r for r in rooms if not r.properties.get("explored")]
        explored_rooms = [r for r in rooms if r.properties.get("explored")]

        if unexplored:
            lines.append(f"UNEXPLORED ({len(unexplored)}):")
            for room in unexplored[:10]:
                objs = self.get_objects_in_room(room.id)
                obj_strs = [f"{o.name}({o.certainty:.0%})" for o in objs]

                # Connected rooms with distance
                connections = []
                walls = []
                for e in self.edges:
                    if e.relation == "connected_to" and (e.source == room.id or e.target == room.id):
                        other = e.target if e.source == room.id else e.source
                        connections.append(f"{other}({e.distance:.0f}px)")
                    elif e.relation == "separated_by_wall" and (e.source == room.id or e.target == room.id):
                        other = e.target if e.source == room.id else e.source
                        walls.append(other)

                fidx = room.properties.get('frontier_idx', '?')
                line = f"  {room.id} [frontier={fidx}]: type={room.name}, certainty={room.certainty:.0%}"
                if obj_strs:
                    line += f"\n    objects: {', '.join(obj_strs)}"

                    # Object spatial relations
                    for obj in objs:
                        obj_edges = [e for e in self.edges
                                     if (e.source == obj.id or e.target == obj.id)
                                     and e.relation in ("next_to", "near")]
                        for oe in obj_edges[:2]:
                            other_id = oe.target if oe.source == obj.id else oe.source
                            other = self.nodes.get(other_id)
                            if other:
                                line += f"\n    {obj.name} ──{oe.relation}({oe.distance:.0f}px)──> {other.name}"

                if connections:
                    line += f"\n    connected: {', '.join(connections[:4])}"
                if walls:
                    line += f"\n    walls: {', '.join(walls[:3])}"
                lines.append(line)

        if explored_rooms:
            lines.append(f"\nEXPLORED ({len(explored_rooms)}):")
            for room in explored_rooms[:6]:
                objs = self.get_objects_in_room(room.id)
                obj_names = [o.name for o in objs]
                lines.append(f"  {room.id}: type={room.name}, objects=[{','.join(obj_names)}]")

        return "\n".join(lines)


# ═══════════════════════════════════════
# KG UPDATER
# ═══════════════════════════════════════

ROOM_HINTS = {
    "toilet": "bathroom", "sink": "bathroom", "bathtub": "bathroom",
    "shower": "bathroom", "towel": "bathroom",
    "bed": "bedroom", "chest_of_drawers": "bedroom",
    "sofa": "living_room", "tv_monitor": "living_room", "fireplace": "living_room",
    "table": "kitchen", "chair": "living_room", "plant": "living_room",
    "stairs": "hallway",
}

MERGE_DISTANCE = 30  # pixels — same object if within this distance


class KGUpdater:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._grid_size = 50

    def _pos_to_room_id(self, y, x):
        gy = int(y // self._grid_size)
        gx = int(x // self._grid_size)
        return f"room_{gy}_{gx}"

    def _find_or_create_object(self, category, pos, confidence):
        """Find existing object node nearby, or create new one. Merge if close."""
        # Search for existing object of same category within MERGE_DISTANCE
        for node in self.kg.get_nodes_by_type("object"):
            if node.properties.get("category") != category:
                continue
            dist = np.sqrt((node.position[0] - pos[0])**2 +
                          (node.position[1] - pos[1])**2)
            if dist < MERGE_DISTANCE:
                # Merge: update certainty and position
                old_cert = node.certainty
                node.certainty = 1.0 - (1.0 - old_cert) * (1.0 - confidence)
                node.certainty = min(node.certainty, 0.99)
                # Running average position
                node.position = (
                    0.7 * node.position[0] + 0.3 * pos[0],
                    0.7 * node.position[1] + 0.3 * pos[1],
                )
                count = node.properties.get("detection_count", 1) + 1
                node.properties["detection_count"] = count
                return node.id, False  # existing node

        # Create new object node
        room_id = self._pos_to_room_id(pos[0], pos[1])
        obj_id = f"{category}_{room_id}"
        # If this ID exists but is far, append a counter
        if obj_id in self.kg.nodes:
            counter = 2
            while f"{obj_id}_{counter}" in self.kg.nodes:
                counter += 1
            obj_id = f"{obj_id}_{counter}"

        self.kg.add_node(KGNode(
            id=obj_id, node_type="object", name=category,
            certainty=confidence, position=pos,
            properties={"category": category, "detection_count": 1}
        ))
        return obj_id, True  # new node

    def update(self, enriched_frontiers, object_list, pose_pred,
               wall_list, full_map_pred, target_name):
        """Update KG from current map state."""

        # ── Robots ──
        for i, pos in enumerate(pose_pred):
            robot_id = f"robot_{i}"
            self.kg.add_node(KGNode(
                id=robot_id, node_type="robot", name=robot_id,
                certainty=1.0, position=(pos[0], pos[1])
            ))

            # Robot's current room
            room_id = self._pos_to_room_id(pos[0], pos[1])
            self.kg.add_node(KGNode(
                id=room_id, node_type="room", name="explored",
                certainty=0.8, position=(pos[0], pos[1]),
                properties={"explored": True,
                           "observation_count": self.kg.nodes.get(room_id, KGNode("","","",0)).properties.get("observation_count", 0) + 1}
            ))
            self.kg.add_edge(KGEdge(robot_id, room_id, "in"))
            self.kg.add_edge(KGEdge(robot_id, room_id, "explored"))

            # Path distances to unexplored rooms
            for room in self.kg.get_nodes_by_type("room"):
                if room.properties.get("explored"):
                    continue
                d = np.sqrt((pos[0] - room.position[0])**2 + (pos[1] - room.position[1])**2)
                if d < 300:
                    self.kg.add_edge(KGEdge(robot_id, room.id, "path_to", distance=d))

        # ── Frontier rooms (unexplored areas) ──
        for ef in enriched_frontiers:
            cy, cx = ef['centroid'][0], ef['centroid'][1]
            room_id = self._pos_to_room_id(cy, cx)
            nearby = ef.get('nearby_objects', [])

            enriched_room_type = ef.get("room_type", "unknown") or "unknown"
            if enriched_room_type.startswith("likely_"):
                room_type = enriched_room_type
                enriched_room_type = enriched_room_type[len("likely_"):]
            elif enriched_room_type != "unknown":
                room_type = f"likely_{enriched_room_type}"
            else:
                room_type = "unknown"
            try:
                room_confidence = float(ef.get("room_confidence", 0.0))
            except (TypeError, ValueError):
                room_confidence = 0.0
            if not np.isfinite(room_confidence):
                room_confidence = 0.0
            room_confidence = max(0.0, min(room_confidence, 0.99))

            is_explored = self.kg.nodes.get(room_id, KGNode("","","",0)).properties.get("explored", False)
            cert = 0.1 if enriched_room_type == "unknown" else max(0.1, room_confidence)
            if is_explored:
                cert = 0.8

            self.kg.add_node(KGNode(
                id=room_id, node_type="room", name=room_type,
                certainty=cert, position=(cy, cx),
                properties={"frontier_idx": ef['idx'], "size": ef['area'],
                           "explored": is_explored,
                           "room_type": enriched_room_type,
                           "room_confidence": room_confidence,
                           "target_prior": ef.get("target_prior", 0.0),
                           "second_room": ef.get("second_room", "unknown"),
                           "room_margin": ef.get("room_margin", 0.0),
                           "room_scores": ef.get("room_scores", {})}
            ))

            # Add nearby objects with real confidence (from enriched data)
            for obj_name in nearby:
                obj_pos = (cy, cx)  # approximate position near frontier
                obj_id, is_new = self._find_or_create_object(obj_name, obj_pos, 0.7)
                self.kg.add_edge(KGEdge(room_id, obj_id, "contains"))
                if obj_name in ROOM_HINTS:
                    self.kg.add_edge(KGEdge(obj_id, f"roomtype_{ROOM_HINTS[obj_name]}", "suggests"))

        # ── Objects from Detectron2 (with real confidence) ──
        if object_list:
            for obj_name, positions in object_list.items():
                for pos_data in positions[:5]:
                    try:
                        coords = pos_data[0][0] if len(pos_data) > 0 else (0, 0)
                        obj_pos = (float(coords[0]), float(coords[1]))
                    except:
                        continue
                    if obj_pos == (0, 0):
                        continue

                    # Use sem_pred_prob_thr as default confidence (0.9 in Co-NavGPT)
                    confidence = 0.9

                    obj_id, is_new = self._find_or_create_object(obj_name, obj_pos, confidence)
                    room_id = self._pos_to_room_id(obj_pos[0], obj_pos[1])
                    self.kg.add_edge(KGEdge(room_id, obj_id, "contains"))

                    if obj_name in ROOM_HINTS:
                        self.kg.add_edge(KGEdge(obj_id, f"roomtype_{ROOM_HINTS[obj_name]}", "suggests"))

            # ── Spatial edges between objects in same room ──
            objects = self.kg.get_nodes_by_type("object")
            for i, o1 in enumerate(objects):
                for j, o2 in enumerate(objects):
                    if i >= j:
                        continue
                    if o1.position == (0, 0) or o2.position == (0, 0):
                        continue
                    dist = np.sqrt((o1.position[0] - o2.position[0])**2 +
                                  (o1.position[1] - o2.position[1])**2)
                    if dist < 20:
                        self.kg.add_edge(KGEdge(o1.id, o2.id, "next_to", distance=dist))
                    elif dist < 60:
                        self.kg.add_edge(KGEdge(o1.id, o2.id, "near", distance=dist))

        # ── Room connections with distance ──
        rooms = self.kg.get_nodes_by_type("room")
        for i, r1 in enumerate(rooms):
            for j, r2 in enumerate(rooms):
                if i >= j:
                    continue
                dist = np.sqrt((r1.position[0] - r2.position[0])**2 +
                              (r1.position[1] - r2.position[1])**2)
                if dist < 150:
                    wall_key = (r1.id, r2.id, "separated_by_wall")
                    if wall_key not in self.kg._edge_set:
                        self.kg.add_edge(KGEdge(r1.id, r2.id, "connected_to", distance=dist))

        # ── Walls ──
        if wall_list is not None and len(wall_list) > 0:
            for wall in wall_list:
                try:
                    coords = wall[0]
                    wall_mid_y = (coords[0] + coords[2]) / 2
                    wall_mid_x = (coords[1] + coords[3]) / 2
                    wall_len = np.sqrt((coords[2]-coords[0])**2 + (coords[3]-coords[1])**2)
                    if wall_len > 10:
                        dy = coords[2] - coords[0]
                        dx = coords[3] - coords[1]
                        norm_y, norm_x = -dx / wall_len * 30, dy / wall_len * 30
                        side_a = self._pos_to_room_id(wall_mid_y + norm_y, wall_mid_x + norm_x)
                        side_b = self._pos_to_room_id(wall_mid_y - norm_y, wall_mid_x - norm_x)
                        if side_a != side_b:
                            self.kg.add_edge(KGEdge(side_a, side_b, "separated_by_wall"))
                except:
                    pass

        # ── Check target on map ──
        if target_name and full_map_pred is not None:
            from constants import hm3d_category
            import torch
            sem = full_map_pred[4:]
            for i, cat in enumerate(hm3d_category):
                if cat == target_name and i < sem.shape[0]:
                    count = (sem[i] > 0.1).sum().item()
                    if count > 5:
                        ys, xs = torch.where(sem[i] > 0.1)
                        target_pos = (int(ys.float().mean()), int(xs.float().mean()))
                        self.kg.add_node(KGNode(
                            id=f"TARGET_{target_name}",
                            node_type="object", name=f"TARGET:{target_name}",
                            certainty=0.95, position=target_pos,
                            properties={"category": target_name, "is_target": True}
                        ))
