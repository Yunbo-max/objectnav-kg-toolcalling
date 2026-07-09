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

try:
    from constants import mp3d_context_category
except ImportError:
    mp3d_context_category = []

CONTEXT_ONLY_CATEGORIES = set(mp3d_context_category)


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
    def __init__(self, certainty_cap: float = 0.99):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self._edge_set: Set[Tuple[str, str, str]] = set()
        self.certainty_cap = certainty_cap

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self._edge_set.clear()

    def add_node(self, node: KGNode):
        if node.id in self.nodes:
            existing = self.nodes[node.id]
            # Merge certainty: 1 - (1-old)(1-new)
            existing.certainty = 1.0 - (1.0 - existing.certainty) * (1.0 - node.certainty)
            existing.certainty = min(existing.certainty, self.certainty_cap)
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

    def remove_edge(self, source: str, target: str, relation: str):
        key = (source, target, relation)
        if key not in self._edge_set:
            return
        self.edges = [
            e for e in self.edges
            if (e.source, e.target, e.relation) != key
        ]
        self._edge_set.discard(key)

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
        unexplored = [
            r for r in rooms
            if not r.properties.get("explored") and r.properties.get("active_frontier")
        ]
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
    "toilet": "bathroom",
    "sink": "bathroom",
    "bathtub": "bathroom",
    "shower": "bathroom",
    "towel": "bathroom",
    "bed": "bedroom",
    "chest_of_drawers": "bedroom",
    "clothes": "bedroom",
    "cushion": "bedroom",
    "sofa": "living_room",
    "tv_monitor": "living_room",
    "fireplace": "living_room",
    "picture": "living_room",
    "seating": "living_room",
    "table": "kitchen_or_dining",
    "chair": "dining_or_living",
    "counter": "kitchen",
    "cabinet": "kitchen",
    "refrigerator": "kitchen",
    "appliances": "kitchen",
    "plant": "living_room",
    "stairs": "hallway",
    "stool": "kitchen_or_bar",
    "gym_equipment": "gym",
}

MERGE_DISTANCE = 30  # pixels — same object if within this distance


def _semantic_categories_for_map(full_map_pred, semantic_categories=None):
    if semantic_categories is not None:
        return list(semantic_categories)
    try:
        from constants import HM3D_SEMANTIC_CATEGORIES
    except ImportError:
        return []
    return HM3D_SEMANTIC_CATEGORIES


class KGUpdater:
    def __init__(
        self,
        kg: KnowledgeGraph,
        room_grid_scale: int = 50,
        merge_radius_px: float = MERGE_DISTANCE,
        object_merge_enabled: bool = True,
        object_merge_radius_px: Optional[float] = None,
        next_to_px: float = 20,
        near_px: float = 60,
        room_connect_px: float = 150,
        max_room_connections: int = 3,
        create_pseudo_doors: bool = False,
        suggests_as_property: bool = True,
        certainty_cap: Optional[float] = None,
    ):
        self.kg = kg
        self._grid_size = int(room_grid_scale)
        self.merge_radius_px = float(merge_radius_px)
        self.object_merge_enabled = bool(object_merge_enabled)
        self.object_merge_radius_px = float(
            object_merge_radius_px if object_merge_radius_px is not None else merge_radius_px
        )
        self.next_to_px = float(next_to_px)
        self.near_px = float(near_px)
        self.room_connect_px = float(room_connect_px)
        self.max_room_connections = int(max_room_connections) if max_room_connections is not None else 0
        self.create_pseudo_doors = bool(create_pseudo_doors)
        self.suggests_as_property = bool(suggests_as_property)
        self.certainty_cap = float(certainty_cap if certainty_cap is not None else kg.certainty_cap)

    def _pos_to_room_id(self, y, x):
        gy = int(y // self._grid_size)
        gx = int(x // self._grid_size)
        return f"room_{gy}_{gx}"

    def _room_id_center(self, room_id):
        try:
            _, gy, gx = room_id.split("_", 2)
            return ((int(gy) + 0.5) * self._grid_size, (int(gx) + 0.5) * self._grid_size)
        except Exception:
            return (0, 0)

    def _ensure_region_node(self, room_id, position=None, **properties):
        if room_id in self.kg.nodes:
            return
        pos = position if position is not None else self._room_id_center(room_id)
        props = {"explored": False, "active_frontier": False, "inferred": True}
        props.update(properties)
        self.kg.add_node(KGNode(
            id=room_id,
            node_type="room",
            name="unknown",
            certainty=0.1,
            position=pos,
            properties=props,
        ))

    def _clamp_confidence(self, value, default=0.5):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = default
        if not np.isfinite(confidence) or confidence <= 0:
            confidence = default
        return max(0.0, min(confidence, self.certainty_cap))

    def _object_contour_position_confidence(
        self,
        obj_name,
        contour,
        full_map_pred,
        semantic_categories=None,
    ):
        pts = np.asarray(contour).reshape(-1, 2)
        if pts.size == 0:
            return (0, 0), 0.0

        xs = pts[:, 0].astype(float)
        ys = pts[:, 1].astype(float)
        obj_pos = (float(ys.mean()), float(xs.mean()))

        if full_map_pred is None:
            return obj_pos, self._clamp_confidence(0.9)

        categories = _semantic_categories_for_map(full_map_pred, semantic_categories)
        try:
            cat_idx = categories.index(obj_name)
        except ValueError:
            return obj_pos, self._clamp_confidence(0.9)

        sem = full_map_pred[4:]
        if cat_idx >= sem.shape[0]:
            return obj_pos, self._clamp_confidence(0.9)

        y_min = max(0, int(np.floor(ys.min())))
        y_max = min(int(sem.shape[1]) - 1, int(np.ceil(ys.max())))
        x_min = max(0, int(np.floor(xs.min())))
        x_max = min(int(sem.shape[2]) - 1, int(np.ceil(xs.max())))
        if y_max < y_min or x_max < x_min:
            return obj_pos, self._clamp_confidence(0.9)

        patch = sem[cat_idx, y_min:y_max + 1, x_min:x_max + 1]
        if hasattr(patch, "detach"):
            patch = patch.detach().cpu().numpy()
        positive = patch[patch > 0]
        confidence = float(positive.max()) if positive.size else 0.9
        return obj_pos, self._clamp_confidence(confidence, default=0.9)

    def _find_or_create_object(self, category, pos, confidence):
        """Find existing object node nearby, or create new one. Merge if close."""
        confidence = self._clamp_confidence(confidence)
        # Search for existing object of same category within MERGE_DISTANCE
        for node in self.kg.get_nodes_by_type("object"):
            if node.properties.get("category") != category:
                continue
            dist = np.sqrt((node.position[0] - pos[0])**2 +
                          (node.position[1] - pos[1])**2)
            if dist < self.merge_radius_px:
                # Merge: update certainty and position
                old_cert = node.certainty
                node.certainty = 1.0 - (1.0 - old_cert) * (1.0 - confidence)
                node.certainty = min(node.certainty, self.certainty_cap)
                if category in CONTEXT_ONLY_CATEGORIES:
                    node.properties["context_only"] = True
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

        properties = {"category": category, "detection_count": 1}
        if category in CONTEXT_ONLY_CATEGORIES:
            properties["context_only"] = True

        self.kg.add_node(KGNode(
            id=obj_id, node_type="object", name=category,
            certainty=confidence, position=pos,
            properties=properties
        ))
        return obj_id, True  # new node

    def _record_roomtype_suggestion(self, obj_id, obj_name):
        if obj_name not in ROOM_HINTS:
            return
        room_type = ROOM_HINTS[obj_name]
        if self.suggests_as_property and obj_id in self.kg.nodes:
            node = self.kg.nodes[obj_id]
            suggestions = set(node.properties.get("suggests_room_types", []))
            suggestions.add(room_type)
            node.properties["suggests_room_types"] = sorted(suggestions)
        else:
            self.kg.add_edge(KGEdge(obj_id, f"roomtype_{room_type}", "suggests"))

    def _clear_inferred_connections(self):
        """Rebuild inferred topology each update so stale dense edges do not accumulate."""
        self.kg.edges = [
            e for e in self.kg.edges
            if e.relation not in ("connected_to", "connected_via")
        ]
        self.kg._edge_set = {
            (e.source, e.target, e.relation)
            for e in self.kg.edges
        }
        for node_id, node in list(self.kg.nodes.items()):
            if node.node_type == "door":
                del self.kg.nodes[node_id]

    def _clear_object_spatial_edges(self):
        """Rebuild object-object proximity edges from the current deduplicated nodes."""
        self.kg.edges = [
            e for e in self.kg.edges
            if e.relation not in ("next_to", "near")
        ]
        self.kg._edge_set = {
            (e.source, e.target, e.relation)
            for e in self.kg.edges
        }

    def _rewrite_edges_after_object_merge(self, replacement: Dict[str, str]):
        if not replacement:
            return

        new_edges = []
        new_edge_set = set()
        for edge in self.kg.edges:
            source = replacement.get(edge.source, edge.source)
            target = replacement.get(edge.target, edge.target)
            if source == target:
                continue
            if source not in self.kg.nodes or target not in self.kg.nodes:
                continue
            key = (source, target, edge.relation)
            if key in new_edge_set:
                continue
            new_edges.append(KGEdge(
                source=source,
                target=target,
                relation=edge.relation,
                distance=edge.distance,
                properties=dict(edge.properties),
            ))
            new_edge_set.add(key)

        self.kg.edges = new_edges
        self.kg._edge_set = new_edge_set

    def _merge_object_cluster(self, cluster: List[KGNode]) -> Tuple[str, List[str]]:
        cluster = sorted(
            cluster,
            key=lambda node: (
                not node.properties.get("is_target", False),
                -node.properties.get("detection_count", 1),
                -node.certainty,
                node.id,
            ),
        )
        keep = cluster[0]
        remove = cluster[1:]
        if not remove:
            return keep.id, []

        total_weight = 0.0
        weighted_y = 0.0
        weighted_x = 0.0
        certainty_product = 1.0
        detection_count = 0
        suggestions = set(keep.properties.get("suggests_room_types", []))

        for node in cluster:
            count = max(1, int(node.properties.get("detection_count", 1)))
            weight = max(0.05, float(node.certainty)) * count
            total_weight += weight
            weighted_y += node.position[0] * weight
            weighted_x += node.position[1] * weight
            certainty_product *= (1.0 - float(node.certainty))
            detection_count += count
            suggestions.update(node.properties.get("suggests_room_types", []))

        if total_weight > 0:
            keep.position = (weighted_y / total_weight, weighted_x / total_weight)
        keep.certainty = min(1.0 - certainty_product, self.certainty_cap)
        keep.properties["detection_count"] = detection_count
        keep.properties["merged_object_count"] = len(cluster)
        keep.properties["merged_from"] = sorted(node.id for node in remove)
        if suggestions:
            keep.properties["suggests_room_types"] = sorted(suggestions)

        for node in remove:
            del self.kg.nodes[node.id]

        return keep.id, [node.id for node in remove]

    def _deduplicate_objects(self):
        """Merge same-category object nodes whose map positions are close.

        This second-stage merge is intentionally separate from detection-time
        merging. It catches duplicates accumulated through frontier priors and
        contour detections across update steps.
        """
        if not self.object_merge_enabled or self.object_merge_radius_px <= 0:
            return

        objects_by_category: Dict[str, List[KGNode]] = {}
        for node in self.kg.get_nodes_by_type("object"):
            if node.properties.get("is_target"):
                continue
            category = node.properties.get("category") or node.name
            objects_by_category.setdefault(category, []).append(node)

        replacement: Dict[str, str] = {}
        for _, objects in objects_by_category.items():
            remaining = set(node.id for node in objects)
            by_id = {node.id: node for node in objects}
            while remaining:
                seed_id = min(remaining)
                seed = by_id[seed_id]
                remaining.remove(seed_id)
                cluster = [seed]

                changed = True
                while changed:
                    changed = False
                    for other_id in list(remaining):
                        other = by_id[other_id]
                        if any(
                            np.sqrt((other.position[0] - member.position[0])**2 +
                                    (other.position[1] - member.position[1])**2)
                            <= self.object_merge_radius_px
                            for member in cluster
                        ):
                            cluster.append(other)
                            remaining.remove(other_id)
                            changed = True

                keep_id, removed_ids = self._merge_object_cluster(cluster)
                for removed_id in removed_ids:
                    replacement[removed_id] = keep_id

        self._rewrite_edges_after_object_merge(replacement)

    def _rebuild_object_spatial_edges(self):
        self._clear_object_spatial_edges()
        objects = self.kg.get_nodes_by_type("object")
        for i, o1 in enumerate(objects):
            if o1.position == (0, 0):
                continue
            for j, o2 in enumerate(objects):
                if i >= j or o2.position == (0, 0):
                    continue
                dist = np.sqrt((o1.position[0] - o2.position[0])**2 +
                               (o1.position[1] - o2.position[1])**2)
                if dist < self.next_to_px:
                    self.kg.add_edge(KGEdge(o1.id, o2.id, "next_to", distance=dist))
                elif dist < self.near_px:
                    self.kg.add_edge(KGEdge(o1.id, o2.id, "near", distance=dist))

    def _rooms_separated_by_wall(self, room_a, room_b):
        return (
            (room_a, room_b, "separated_by_wall") in self.kg._edge_set
            or (room_b, room_a, "separated_by_wall") in self.kg._edge_set
        )

    def update(
        self,
        enriched_frontiers,
        object_list,
        pose_pred,
        wall_list,
        full_map_pred,
        target_name,
        semantic_categories=None,
    ):
        """Update KG from current map state."""

        # Frontier indices are step-local. Clear them before marking the current
        # frontier set so the brain cannot assign robots to stale rooms.
        for room in self.kg.get_nodes_by_type("room"):
            room.properties["active_frontier"] = False

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
            room_confidence = self._clamp_confidence(
                ef.get("room_confidence", 0.0),
                default=0.0,
            )

            is_explored = self.kg.nodes.get(room_id, KGNode("","","",0)).properties.get("explored", False)
            cert = 0.1 if enriched_room_type == "unknown" else max(0.1, room_confidence)
            if is_explored:
                cert = 0.8

            self.kg.add_node(KGNode(
                id=room_id, node_type="room", name=room_type,
                certainty=cert, position=(cy, cx),
                properties={"frontier_idx": ef['idx'], "size": ef['area'],
                           "explored": is_explored,
                           "active_frontier": not is_explored,
                           "room_type": enriched_room_type,
                           "room_confidence": room_confidence,
                           "target_prior": ef.get("target_prior", 0.0),
                           "second_room": ef.get("second_room", "unknown"),
                           "room_margin": ef.get("room_margin", 0.0),
                           "room_scores": ef.get("room_scores", {})}
            ))

            # Add nearby objects with real confidence (from enriched data)
            nearby_scores = ef.get('nearby_object_scores', {})
            for obj_name in nearby:
                obj_pos = (cy, cx)  # approximate position near frontier
                confidence = nearby_scores.get(obj_name, 0.7)
                obj_id, is_new = self._find_or_create_object(obj_name, obj_pos, confidence)
                self.kg.add_edge(KGEdge(room_id, obj_id, "contains"))
                self._record_roomtype_suggestion(obj_id, obj_name)

        # ── Objects from Detectron2 (with real confidence) ──
        if object_list:
            for obj_name, positions in object_list.items():
                for pos_data in positions[:5]:
                    try:
                        obj_pos, confidence = self._object_contour_position_confidence(
                            obj_name,
                            pos_data,
                            full_map_pred,
                            semantic_categories,
                        )
                    except Exception:
                        continue
                    if obj_pos == (0, 0):
                        continue

                    obj_id, is_new = self._find_or_create_object(obj_name, obj_pos, confidence)
                    room_id = self._pos_to_room_id(obj_pos[0], obj_pos[1])
                    self._ensure_region_node(room_id, position=obj_pos, inferred_from="object")
                    self.kg.add_edge(KGEdge(room_id, obj_id, "contains"))
                    self._record_roomtype_suggestion(obj_id, obj_name)

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
                            if side_a in self.kg.nodes and side_b in self.kg.nodes:
                                self.kg.add_edge(KGEdge(side_a, side_b, "separated_by_wall"))
                                self.kg.remove_edge(side_a, side_b, "connected_to")
                                self.kg.remove_edge(side_b, side_a, "connected_to")
                except:
                    pass

        # ── Room connections with distance ──
        self._clear_inferred_connections()
        rooms = self.kg.get_nodes_by_type("room")
        candidates = []
        for i, r1 in enumerate(rooms):
            for j, r2 in enumerate(rooms):
                if i >= j:
                    continue
                if self._rooms_separated_by_wall(r1.id, r2.id):
                    continue
                dist = np.sqrt((r1.position[0] - r2.position[0])**2 +
                               (r1.position[1] - r2.position[1])**2)
                if dist < self.room_connect_px:
                    candidates.append((float(dist), r1, r2))

        degree = {room.id: 0 for room in rooms}
        for dist, r1, r2 in sorted(candidates, key=lambda item: item[0]):
            if self.max_room_connections > 0:
                if degree.get(r1.id, 0) >= self.max_room_connections:
                    continue
                if degree.get(r2.id, 0) >= self.max_room_connections:
                    continue
            self.kg.add_edge(KGEdge(r1.id, r2.id, "connected_to", distance=dist))
            degree[r1.id] = degree.get(r1.id, 0) + 1
            degree[r2.id] = degree.get(r2.id, 0) + 1

            if self.create_pseudo_doors:
                door_id = f"door_{r1.id}_{r2.id}"
                door_pos = (
                    (r1.position[0] + r2.position[0]) / 2.0,
                    (r1.position[1] + r2.position[1]) / 2.0,
                )
                self.kg.add_node(KGNode(
                    id=door_id, node_type="door", name="door",
                    certainty=0.6, position=door_pos,
                    properties={"rooms": [r1.id, r2.id], "inferred": True}
                ))
                self.kg.add_edge(KGEdge(r1.id, door_id, "connected_via", distance=dist / 2.0))
                self.kg.add_edge(KGEdge(door_id, r2.id, "connected_via", distance=dist / 2.0))

        # ── Check target on map ──
        if target_name and full_map_pred is not None:
            import torch
            sem = full_map_pred[4:]
            for i, cat in enumerate(_semantic_categories_for_map(full_map_pred, semantic_categories)):
                if cat == target_name and i < sem.shape[0]:
                    count = (sem[i] > 0.1).sum().item()
                    if count > 5:
                        ys, xs = torch.where(sem[i] > 0.1)
                        confidence = self._clamp_confidence(sem[i][ys, xs].max().item(), default=0.95)
                        target_pos = (int(ys.float().mean()), int(xs.float().mean()))
                        self.kg.add_node(KGNode(
                            id=f"TARGET_{target_name}",
                            node_type="object", name=f"TARGET:{target_name}",
                            certainty=confidence, position=target_pos,
                            properties={"category": target_name, "is_target": True}
                        ))

        self._deduplicate_objects()
        self._rebuild_object_spatial_edges()
