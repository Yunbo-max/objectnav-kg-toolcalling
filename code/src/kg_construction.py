"""MindNav persistent evidence graph and current-frontier view.

The graph keeps episode-level region/object evidence, while executable frontier
indices live only in :class:`CurrentFrontierView`. Scene memory stores graph
geometry in Habitat world ``(x, z)`` coordinates; the episode ablation retains
map ``(row, column)`` coordinates.
"""
import json
import math

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


MAP_SIZE = 480
ROOM_CELL_M = 2.5
UNDIRECTED_RELATIONS = {
    "connected_to",
    "separated_by_wall",
    "next_to",
    "near",
}


@dataclass
class KGNode:
    id: str
    node_type: str          # "room", "object", "robot", "door"
    name: str               # "bathroom", "sink", "robot_0", "door_1"
    certainty: float        # from detections: 1 - ∏(1-conf_i)
    position: Tuple[float, float] = (0, 0)
    properties: Dict = field(default_factory=dict)
    # For rooms: {"explored": bool, "observation_count": int}
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


@dataclass(frozen=True)
class EpisodeFrame:
    """Rigid transform between the episode-centered map and HM3D world.

    The mapper stores row/column coordinates in a 24 m square centered at the
    episode start.  Habitat's episodic GPS uses a start-relative, rotated
    frame, so the two horizontal axes are inverted before applying the
    episode-start quaternion.
    """

    scene_id: str
    start_position: Tuple[float, float, float]
    start_rotation: Tuple[float, float, float, float]  # Habitat x, y, z, w
    map_size_px: int = MAP_SIZE
    map_resolution_m: float = 0.05

    @property
    def floor_id(self) -> int:
        return int(round(float(self.start_position[1]) / 0.5))

    def map_rc_to_world_xz(self, row: float, col: float) -> Tuple[float, float]:
        center = float(self.map_size_px) / 2.0
        # See EpisodicGPSSensor and LLM_Agent.get_pose_change.  Map row/col
        # offsets are (-episode_x, -episode_z), respectively.
        local_x = -(float(row) - center) * self.map_resolution_m
        local_z = -(float(col) - center) * self.map_resolution_m
        qx, qy, qz, qw = (float(value) for value in self.start_rotation)
        # Quaternion rotation q * (x, 0, z) * q^-1 without a SciPy runtime
        # dependency.  Habitat stores coefficients as x, y, z, w.
        vx, vy, vz = local_x, 0.0, local_z
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        rz = vz + qw * tz + (qx * ty - qy * tx)
        return (
            float(self.start_position[0]) + rx,
            float(self.start_position[2]) + rz,
        )

    def room_id_from_map_rc(self, row: float, col: float) -> str:
        world_x, world_z = self.map_rc_to_world_xz(row, col)
        grid_x = math.floor(world_x / ROOM_CELL_M)
        grid_z = math.floor(world_z / ROOM_CELL_M)
        return (
            f"scene_room_f{self.floor_id}_x{grid_x}_z{grid_z}"
        )


@dataclass(frozen=True)
class CurrentFrontier:
    """One executable frontier candidate for the current planning step only."""

    idx: int
    room_id: str
    centroid: Tuple[float, float]
    area: float
    target_prior: float = 0.0
    room_type: str = "unknown"
    room_confidence: float = 0.0
    world_position_xz: Optional[Tuple[float, float]] = None
    floor_id: Optional[int] = None


@dataclass
class CurrentFrontierView:
    """Ephemeral one-to-many mapping between persistent rooms and frontiers.

    Frontier indices are regenerated on every planning step, so this view must
    never be stored in the persistent knowledge graph.
    """

    frontiers: Dict[int, CurrentFrontier]
    room_to_frontiers: Dict[str, Tuple[int, ...]]

    @classmethod
    def from_enriched(cls, enriched_frontiers, room_id_resolver=None,
                      world_position_resolver=None, grid_size=50):
        if room_id_resolver is None:
            def room_id_resolver(y, x):
                return f"room_{int(y // grid_size)}_{int(x // grid_size)}"

        frontiers = {}
        room_to_frontiers = {}
        for enriched in enriched_frontiers:
            idx = int(enriched["idx"])
            if idx in frontiers:
                raise ValueError(f"Duplicate current frontier index: {idx}")

            centroid = enriched.get("centroid")
            if centroid is None or len(centroid) != 2:
                raise ValueError(
                    f"Frontier {idx} must have a two-dimensional centroid"
                )
            cy, cx = float(centroid[0]), float(centroid[1])
            room_id = room_id_resolver(cy, cx)
            world_position_xz = None
            floor_id = None
            if world_position_resolver is not None:
                world_position_xz, floor_id = world_position_resolver(cy, cx)
            frontier = CurrentFrontier(
                idx=idx,
                room_id=room_id,
                centroid=(cy, cx),
                area=float(enriched.get("area", 0.0)),
                target_prior=float(
                    enriched.get("target_prior",
                                 enriched.get("python_prior", 0.0))
                ),
                room_type=str(enriched.get("room_type", "unknown")),
                room_confidence=float(enriched.get("room_confidence", 0.0)),
                world_position_xz=world_position_xz,
                floor_id=floor_id,
            )
            frontiers[idx] = frontier
            room_to_frontiers.setdefault(room_id, []).append(idx)

        return cls(
            frontiers=frontiers,
            room_to_frontiers={
                room_id: tuple(indices)
                for room_id, indices in room_to_frontiers.items()
            },
        )

    @property
    def room_ids(self):
        return set(self.room_to_frontiers)

    @staticmethod
    def robot_pose_to_map_rc(pose, map_size=MAP_SIZE):
        """Convert the runtime display-frame ``(x, y)`` pose to map ``(r, c)``.

        Frontier centroids and semantic-map components use matrix coordinates,
        while ``exp_main_brain`` builds robot poses for the vertically flipped
        480x480 visualization.  Keeping this conversion here prevents distance
        comparisons from silently mixing the two frames.
        """
        if pose is None or len(pose) < 2:
            return None
        return (
            float(map_size) - float(pose[1]),
            float(pose[0]),
        )

    def select_frontier(self, room_id=None, robot_position=None,
                        excluded=None):
        """Select a valid frontier using prior, distance, area, then index."""
        excluded = set() if excluded is None else set(excluded)
        if room_id is None:
            candidate_indices = list(self.frontiers)
        else:
            candidate_indices = list(self.room_to_frontiers.get(room_id, ()))

        candidate_indices = [
            idx for idx in candidate_indices if idx not in excluded
        ]
        if not candidate_indices:
            return None

        def rank(idx):
            frontier = self.frontiers[idx]
            distance = 0.0
            if robot_position is not None and len(robot_position) >= 2:
                distance = float(np.hypot(
                    frontier.centroid[0] - float(robot_position[0]),
                    frontier.centroid[1] - float(robot_position[1]),
                ))
            return (
                distance,
                -frontier.area,
                frontier.idx,
            )

        return min(candidate_indices, key=rank)

    def deterministic_assignments(self, pose_pred, num_agents):
        """Return current-only fallback assignments for every robot."""
        if not self.frontiers:
            return {}

        assignments = {}
        used = set()
        for robot_index in range(num_agents):
            pose_value = (
                pose_pred[robot_index]
                if robot_index < len(pose_pred) else None
            )
            position = self.robot_pose_to_map_rc(pose_value)
            frontier_idx = self.select_frontier(
                robot_position=position,
                excluded=used,
            )
            if frontier_idx is None:
                # Reuse is only allowed when there are fewer frontiers than
                # robots. It is still guaranteed to be a current frontier.
                frontier_idx = self.select_frontier(
                    robot_position=position,
                )
            assignments[f"robot_{robot_index}"] = frontier_idx
            used.add(frontier_idx)
        return assignments


class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self._edge_set: Set[Tuple[str, str, str]] = set()
        # Monotonic reset marker lets the decision brain keep recent
        # assignment history strictly episode-local, even if two episodes
        # reach their first planning decision at the same local step.
        self.reset_generation = 0

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self._edge_set.clear()
        self.reset_generation += 1

    def reset_episode_state(self):
        """Drop execution-time state while preserving static scene evidence."""
        robot_ids = {
            node_id for node_id, node in self.nodes.items()
            if node.node_type == "robot"
        }
        for node_id in robot_ids:
            self.nodes.pop(node_id, None)
        dynamic_relations = {"in", "explored", "path_to"}
        self.remove_edges(
            lambda edge: (
                edge.relation in dynamic_relations
                or edge.source in robot_ids
                or edge.target in robot_ids
            )
        )
        for room in self.get_nodes_by_type("room"):
            room.properties["explored"] = False
            for property_name in TRANSIENT_ROOM_PROPERTIES:
                room.properties.pop(property_name, None)
        self.reset_generation += 1

    @staticmethod
    def _edge_key(edge):
        source, target = edge.source, edge.target
        if edge.relation in UNDIRECTED_RELATIONS and target < source:
            source, target = target, source
        return source, target, edge.relation

    def add_node(self, node: KGNode):
        if node.id in self.nodes:
            existing = self.nodes[node.id]
            # Evidence aggregation is node-type specific.  In particular,
            # object noisy-OR is applied only after observation de-duplication
            # in ``_find_or_create_object``; a generic upsert must not count a
            # cumulative map snapshot again.
            existing.certainty = max(
                float(existing.certainty), float(node.certainty)
            )
            if node.node_type == "robot":
                existing.certainty = node.certainty
            if node.name and node.name != "explored":
                existing.name = node.name
            if node.position != (0, 0):
                if node.node_type == "robot":
                    existing.position = node.position
                elif existing.position != (0, 0):
                    existing.position = (
                        0.7 * existing.position[0] + 0.3 * node.position[0],
                        0.7 * existing.position[1] + 0.3 * node.position[1],
                    )
                else:
                    existing.position = node.position
            previous_episode_ids = set(
                existing.properties.get("episode_ids", [])
            )
            incoming_episode_ids = node.properties.get("episode_ids")
            existing.properties.update(node.properties)
            if incoming_episode_ids is not None:
                episode_ids = previous_episode_ids
                episode_ids.update(str(value) for value in incoming_episode_ids)
                existing.properties["episode_ids"] = sorted(episode_ids)
                existing.properties["episode_support_count"] = len(episode_ids)
        else:
            self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge):
        key = self._edge_key(edge)
        if key not in self._edge_set:
            self.edges.append(edge)
            self._edge_set.add(key)
        else:
            # Update distance if edge exists
            for e in self.edges:
                if self._edge_key(e) == key:
                    e.distance = edge.distance
                    previous_episode_ids = set(
                        e.properties.get("episode_ids", [])
                    )
                    incoming_episode_ids = edge.properties.get("episode_ids")
                    e.properties.update(edge.properties)
                    if incoming_episode_ids is not None:
                        episode_ids = previous_episode_ids
                        episode_ids.update(
                            str(value) for value in incoming_episode_ids
                        )
                        e.properties["episode_ids"] = sorted(episode_ids)
                        e.properties["episode_support_count"] = len(episode_ids)
                    break

    def remove_edges(self, predicate):
        """Remove edges matching ``predicate`` and rebuild the key index."""
        self.edges = [edge for edge in self.edges if not predicate(edge)]
        self._edge_set = {self._edge_key(edge) for edge in self.edges}

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

    @staticmethod
    def _snapshot_json_value(value):
        """Convert KG evidence values to deterministic JSON-compatible data."""
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            return [KnowledgeGraph._snapshot_json_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): KnowledgeGraph._snapshot_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [KnowledgeGraph._snapshot_json_value(item) for item in value]
        if isinstance(value, set):
            return [
                KnowledgeGraph._snapshot_json_value(item)
                for item in sorted(value, key=repr)
            ]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            f"Unsupported KG snapshot value: {type(value).__name__}"
        )

    def to_snapshot_dict(self) -> Dict:
        """Return the complete persistent graph without LLM-view truncation.

        Unlike :meth:`to_json`, this method is an analysis/audit serializer. It
        preserves every persistent node, edge, property, and full-precision
        geometry, and therefore must not be used as an LLM prompt projection.
        """
        nodes = []
        for node in sorted(self.nodes.values(), key=lambda item: item.id):
            coordinate_frame = node.properties.get("coordinate_frame", "local")
            local_position = node.properties.get("position_rc")
            if local_position is None and coordinate_frame == "local":
                local_position = node.position
            world_position = node.properties.get("position_world_xz")
            if world_position is None and coordinate_frame == "world":
                world_position = node.position
            nodes.append({
                "id": node.id,
                "node_type": node.node_type,
                "name": node.name,
                "certainty": float(node.certainty),
                # Both are explicit in world mode; never relabel metres as
                # pixels in the audit format.
                "position_rc": (
                    [float(local_position[0]), float(local_position[1])]
                    if local_position is not None else None
                ),
                "position_world_xz": (
                    [float(world_position[0]), float(world_position[1])]
                    if world_position is not None else None
                ),
                "floor_id": node.properties.get("floor_id"),
                "episode_ids": list(node.properties.get("episode_ids", [])),
                "episode_support_count": int(
                    node.properties.get("episode_support_count", 0)
                ),
                "properties": self._snapshot_json_value(node.properties),
            })

        edges = []
        for edge in sorted(
            self.edges,
            key=lambda item: (
                item.source, item.target, item.relation, float(item.distance)
            ),
        ):
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "distance": float(edge.distance),
                "properties": self._snapshot_json_value(edge.properties),
            })

        update_ids = [
            int(properties[key])
            for properties in (
                item.get("properties", {}) for item in nodes + edges
            )
            for key in ("last_seen_update", "first_seen_update")
            if isinstance(properties.get(key), (int, float))
        ]
        return {
            "kg_schema_version": 1,
            "reset_generation": int(self.reset_generation),
            "update_id": max(update_ids, default=0),
            "nodes": nodes,
            "edges": edges,
        }

    def to_text(self, current_frontier_view=None,
                max_historical_rooms=6) -> str:
        """Render persistent context and current assignable rooms separately."""
        lines = []
        world_coordinates = any(
            node.properties.get("coordinate_frame") == "world"
            for node in self.nodes.values()
        )
        distance_unit = "m" if world_coordinates else "px"

        def format_distance(value):
            precision = 1 if world_coordinates else 0
            return f"{value:.{precision}f}{distance_unit}"

        current_room_ids = (
            current_frontier_view.room_ids
            if current_frontier_view is not None else set()
        )

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
                        is_current = (
                            current_frontier_view is None
                            or e.target in current_room_ids
                        )
                        if (target and is_current
                                and not target.properties.get("explored")):
                            paths.append((e.target, e.distance))
                    elif e.relation == "explored":
                        explored.append(e.target)
            paths.sort(key=lambda x: x[1])
            path_str = ", ".join([
                f"{room_id}({format_distance(distance)})"
                for room_id, distance in paths[:5]
            ])
            position_label = (
                "position_world_xz" if world_coordinates else "position_rc"
            )
            position_precision = 2 if world_coordinates else 0
            lines.append(
                f"{r.id}: {position_label}="
                f"({r.position[0]:.{position_precision}f},"
                f"{r.position[1]:.{position_precision}f}), in={in_room}, "
                f"explored={len(explored)} rooms"
            )
            if paths:
                path_label = (
                    "reachable_current" if current_frontier_view is not None
                    else "reachable_unexplored"
                )
                lines.append(f"  {path_label}: {path_str}")

        lines.append("")

        rooms = self.get_nodes_by_type("room")
        unexplored = [
            room for room in rooms if not room.properties.get("explored")
        ]
        explored_rooms = [
            room for room in rooms
            if room.properties.get("explored")
            and room.id not in current_room_ids
        ]

        def format_room(room, current_candidates=None):
            objs = self.get_objects_in_room(room.id)
            obj_strs = [f"{obj.name}({obj.certainty:.0%})" for obj in objs]
            connections = []
            walls = []
            for edge in self.edges:
                touches_room = edge.source == room.id or edge.target == room.id
                if not touches_room:
                    continue
                other = edge.target if edge.source == room.id else edge.source
                if edge.relation == "connected_to":
                    connections.append(
                        f"{other}({format_distance(edge.distance)})"
                    )
                elif edge.relation == "separated_by_wall":
                    walls.append(other)

            candidate_text = ""
            if current_candidates is not None:
                details = []
                for frontier_idx in current_candidates:
                    frontier = current_frontier_view.frontiers[frontier_idx]
                    details.append(
                        f"frontier_{frontier.idx}"
                        f"(area={frontier.area:.0f},"
                        f"centroid=({frontier.centroid[0]:.0f},"
                        f"{frontier.centroid[1]:.0f}))"
                    )
                candidate_text = f" [current_frontiers={','.join(details)}]"

            line = (
                f"  {room.id}{candidate_text}: "
                f"observed_type={room.name or 'unknown'}, "
                f"type_certainty={room.certainty:.0%}, "
                f"observation_count="
                f"{int(room.properties.get('observation_count', 0))}"
            )
            if obj_strs:
                line += f"\n    objects: {', '.join(obj_strs)}"
                for obj in objs:
                    object_edges = [
                        edge for edge in self.edges
                        if (edge.source == obj.id or edge.target == obj.id)
                        and edge.relation in ("next_to", "near")
                    ]
                    for object_edge in object_edges[:2]:
                        other_id = (
                            object_edge.target
                            if object_edge.source == obj.id
                            else object_edge.source
                        )
                        other = self.nodes.get(other_id)
                        if other:
                            line += (
                                f"\n    {obj.name} "
                                f"──{object_edge.relation}"
                                f"({format_distance(object_edge.distance)})"
                                f"──> {other.name}"
                            )
            if connections:
                line += (
                    f"\n    possible_connected: "
                    f"{', '.join(connections[:4])}"
                )
            if walls:
                line += f"\n    walls: {', '.join(walls[:3])}"
            return line

        if current_frontier_view is not None:
            lines.append(
                f"CURRENT ASSIGNABLE ROOMS ({len(current_room_ids)}):"
            )
            for room_id, frontier_indices in sorted(
                    current_frontier_view.room_to_frontiers.items()):
                room = self.nodes.get(room_id)
                if room is None:
                    room = KGNode(
                        id=room_id,
                        node_type="room",
                        name="unknown",
                        certainty=0.0,
                    )
                lines.append(format_room(room, frontier_indices))

            historical = [
                room for room in unexplored if room.id not in current_room_ids
            ]
            historical.sort(
                key=lambda room: (
                    -room.properties.get("last_seen_update", -1),
                    -room.certainty,
                    room.id,
                )
            )
            if historical:
                lines.append(
                    f"\nHISTORICAL ROOM CONTEXT — NOT ASSIGNABLE "
                    f"({len(historical)}):"
                )
                for room in historical[:max_historical_rooms]:
                    lines.append(format_room(room))
        elif unexplored:
            lines.append(f"UNEXPLORED CONTEXT ({len(unexplored)}):")
            for room in unexplored[:10]:
                lines.append(format_room(room))

        if explored_rooms:
            explored_rooms.sort(
                key=lambda room: (
                    -room.properties.get("last_seen_update", -1),
                    room.id,
                )
            )
            lines.append(f"\nEXPLORED ({len(explored_rooms)}):")
            for room in explored_rooms[:6]:
                objs = self.get_objects_in_room(room.id)
                obj_names = [o.name for o in objs]
                lines.append(f"  {room.id}: type={room.name}, objects=[{','.join(obj_names)}]")

        return "\n".join(lines)

    def to_json(self, current_frontier_view=None,
                max_historical_rooms=6) -> str:
        """Render the same bounded KG view as :meth:`to_text` in JSON.

        This serializer intentionally preserves the text representation's
        room selection, relation caps, and displayed numeric precision.  The
        serialization ablation therefore changes the wire format without
        exposing additional graph evidence to the LLM.
        """
        current_room_ids = (
            current_frontier_view.room_ids
            if current_frontier_view is not None else set()
        )
        world_coordinates = any(
            node.properties.get("coordinate_frame") == "world"
            for node in self.nodes.values()
        )
        position_key = (
            "position_world_xz" if world_coordinates else "position_rc"
        )
        distance_key = "distance_m" if world_coordinates else "distance_px"

        def rounded(value):
            return int(round(float(value)))

        def percent(value):
            return int(round(100.0 * float(value)))

        def distance_value(value):
            if world_coordinates:
                return round(float(value), 3)
            return rounded(value)

        robots = []
        for robot in self.get_nodes_by_type("robot"):
            in_room = None
            paths = []
            explored = []
            for edge in self.edges:
                if edge.source != robot.id:
                    continue
                if edge.relation == "in":
                    in_room = edge.target
                elif edge.relation == "path_to":
                    target = self.nodes.get(edge.target)
                    is_current = (
                        current_frontier_view is None
                        or edge.target in current_room_ids
                    )
                    if (target and is_current
                            and not target.properties.get("explored")):
                        paths.append((edge.target, edge.distance))
                elif edge.relation == "explored":
                    explored.append(edge.target)
            paths.sort(key=lambda item: item[1])
            path_label = (
                "reachable_current"
                if current_frontier_view is not None
                else "reachable_unexplored"
            )
            robot_record = {
                "id": robot.id,
                position_key: (
                    [round(float(robot.position[0]), 3),
                     round(float(robot.position[1]), 3)]
                    if world_coordinates else [
                        rounded(robot.position[0]),
                        rounded(robot.position[1]),
                    ]
                ),
                "in_room": in_room,
                "explored_room_count": len(explored),
                path_label: [
                    {
                        "room_id": room_id,
                        distance_key: distance_value(distance),
                    }
                    for room_id, distance in paths[:5]
                ],
            }
            robots.append(robot_record)

        def room_record(room, current_candidates=None):
            objects = self.get_objects_in_room(room.id)
            connections = []
            walls = []
            for edge in self.edges:
                touches_room = (
                    edge.source == room.id or edge.target == room.id
                )
                if not touches_room:
                    continue
                other = (
                    edge.target if edge.source == room.id else edge.source
                )
                if edge.relation == "connected_to":
                    connections.append({
                        "room_id": other,
                        distance_key: distance_value(edge.distance),
                    })
                elif edge.relation == "separated_by_wall":
                    walls.append(other)

            record = {
                "room_id": room.id,
                "observed_type": room.name or "unknown",
                "type_certainty_percent": percent(room.certainty),
                "observation_count": int(
                    room.properties.get("observation_count", 0)
                ),
            }
            if current_candidates is not None:
                record["current_frontiers"] = []
                for frontier_idx in current_candidates:
                    frontier = current_frontier_view.frontiers[frontier_idx]
                    record["current_frontiers"].append({
                        "frontier_id": f"frontier_{frontier.idx}",
                        "area_px": rounded(frontier.area),
                        "centroid_rc": [
                            rounded(frontier.centroid[0]),
                            rounded(frontier.centroid[1]),
                        ],
                    })
            if objects:
                record["objects"] = [
                    {
                        "name": obj.name,
                        "certainty_percent": percent(obj.certainty),
                    }
                    for obj in objects
                ]
                object_relations = []
                for obj in objects:
                    edges = [
                        edge for edge in self.edges
                        if (edge.source == obj.id or edge.target == obj.id)
                        and edge.relation in ("next_to", "near")
                    ]
                    for edge in edges[:2]:
                        other_id = (
                            edge.target if edge.source == obj.id
                            else edge.source
                        )
                        other = self.nodes.get(other_id)
                        if other:
                            object_relations.append({
                                "source": obj.name,
                                "relation": edge.relation,
                                "target": other.name,
                                distance_key: distance_value(edge.distance),
                            })
                if object_relations:
                    record["object_relations"] = object_relations
            if connections:
                record["possible_connected"] = connections[:4]
            if walls:
                record["walls"] = walls[:3]
            return record

        rooms = self.get_nodes_by_type("room")
        unexplored = [
            room for room in rooms if not room.properties.get("explored")
        ]
        explored_rooms = [
            room for room in rooms
            if room.properties.get("explored")
            and room.id not in current_room_ids
        ]
        payload = {
            "format": "mindnav_kg_json_v2",
            "coordinate_frame": (
                "world_xz" if world_coordinates else "map_rc"
            ),
            "robots": robots,
        }

        if current_frontier_view is not None:
            payload["current_assignable_room_count"] = len(current_room_ids)
            current_records = []
            for room_id, frontier_indices in sorted(
                    current_frontier_view.room_to_frontiers.items()):
                room = self.nodes.get(room_id)
                if room is None:
                    room = KGNode(
                        id=room_id,
                        node_type="room",
                        name="unknown",
                        certainty=0.0,
                    )
                current_records.append(room_record(room, frontier_indices))
            payload["current_assignable_rooms"] = current_records

            historical = [
                room for room in unexplored
                if room.id not in current_room_ids
            ]
            historical.sort(
                key=lambda room: (
                    -room.properties.get("last_seen_update", -1),
                    -room.certainty,
                    room.id,
                )
            )
            payload["historical_room_context_not_assignable_count"] = len(
                historical
            )
            payload["historical_room_context_not_assignable"] = [
                room_record(room)
                for room in historical[:max_historical_rooms]
            ]
        elif unexplored:
            payload["unexplored_context"] = [
                room_record(room) for room in unexplored[:10]
            ]

        if explored_rooms:
            explored_rooms.sort(
                key=lambda room: (
                    -room.properties.get("last_seen_update", -1),
                    room.id,
                )
            )
            payload["explored_room_count"] = len(explored_rooms)
            payload["explored_rooms"] = [
                {
                    "room_id": room.id,
                    "observed_type": room.name,
                    "object_names": [
                        obj.name for obj in self.get_objects_in_room(room.id)
                    ],
                }
                for room in explored_rooms[:6]
            ]

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def to_triples(self, current_frontier_view=None,
                   max_historical_rooms=6) -> str:
        """Render the bounded KG view as a list of ``[head, relation, tail]``.

        The triple projection is derived from :meth:`to_json`, so it inherits
        exactly the same room selection, relation caps, and numeric precision.
        Synthetic references only reify attributes such as edge distance; they
        do not expose additional persistent-KG evidence.
        """
        payload = json.loads(self.to_json(
            current_frontier_view=current_frontier_view,
            max_historical_rooms=max_historical_rooms,
        ))
        triples = [[
            "kg",
            "serialization_format",
            "mindnav_kg_triples_v2",
        ]]
        world_coordinates = payload.get("coordinate_frame") == "world_xz"
        distance_key = "distance_m" if world_coordinates else "distance_px"

        def add(head, relation, tail):
            triples.append([head, relation, tail])

        for robot_index, robot in enumerate(payload.get("robots", [])):
            robot_id = robot["id"]
            add(robot_id, "node_type", "robot")
            if world_coordinates:
                add(
                    robot_id, "position_world_x",
                    robot["position_world_xz"][0],
                )
                add(
                    robot_id, "position_world_z",
                    robot["position_world_xz"][1],
                )
            else:
                add(robot_id, "position_row", robot["position_rc"][0])
                add(robot_id, "position_column", robot["position_rc"][1])
            add(robot_id, "in_room", robot["in_room"])
            add(
                robot_id,
                "explored_room_count",
                robot["explored_room_count"],
            )
            path_label = (
                "reachable_current"
                if "reachable_current" in robot
                else "reachable_unexplored"
            )
            for path_index, path in enumerate(robot.get(path_label, [])):
                path_ref = f"{robot_id}::serialized_path_{path_index}"
                add(robot_id, path_label, path_ref)
                add(path_ref, "target_room", path["room_id"])
                add(path_ref, distance_key, path[distance_key])

        def add_room(record, context_role):
            room_id = record["room_id"]
            add(room_id, "node_type", "room")
            add(room_id, "context_role", context_role)
            add(room_id, "observed_type", record["observed_type"])
            if "type_certainty_percent" in record:
                add(
                    room_id,
                    "type_certainty_percent",
                    record["type_certainty_percent"],
                )
            if "observation_count" in record:
                add(
                    room_id,
                    "observation_count",
                    record["observation_count"],
                )

            for frontier in record.get("current_frontiers", []):
                frontier_id = frontier["frontier_id"]
                add(room_id, "has_current_frontier", frontier_id)
                add(frontier_id, "node_type", "frontier")
                add(frontier_id, "area_px", frontier["area_px"])
                add(
                    frontier_id,
                    "centroid_row",
                    frontier["centroid_rc"][0],
                )
                add(
                    frontier_id,
                    "centroid_column",
                    frontier["centroid_rc"][1],
                )

            for object_index, obj in enumerate(record.get("objects", [])):
                object_ref = (
                    f"{room_id}::serialized_object_{object_index}"
                )
                add(room_id, "contains_serialized_object", object_ref)
                add(object_ref, "name", obj["name"])
                add(
                    object_ref,
                    "certainty_percent",
                    obj["certainty_percent"],
                )

            for relation_index, relation in enumerate(
                    record.get("object_relations", [])):
                relation_ref = (
                    f"{room_id}::serialized_object_relation_"
                    f"{relation_index}"
                )
                add(room_id, "has_object_relation", relation_ref)
                add(relation_ref, "source_name", relation["source"])
                add(relation_ref, "relation", relation["relation"])
                add(relation_ref, "target_name", relation["target"])
                add(
                    relation_ref,
                    distance_key,
                    relation[distance_key],
                )

            for connection_index, connection in enumerate(
                    record.get("possible_connected", [])):
                connection_ref = (
                    f"{room_id}::serialized_connection_{connection_index}"
                )
                add(room_id, "possible_connected", connection_ref)
                add(
                    connection_ref,
                    "target_room",
                    connection["room_id"],
                )
                add(
                    connection_ref,
                    distance_key,
                    connection[distance_key],
                )

            for wall_room_id in record.get("walls", []):
                add(room_id, "separated_by_wall", wall_room_id)

            for object_name in record.get("object_names", []):
                add(room_id, "contains_object_name", object_name)

        if "current_assignable_room_count" in payload:
            add(
                "kg",
                "current_assignable_room_count",
                payload["current_assignable_room_count"],
            )
        for room in payload.get("current_assignable_rooms", []):
            add_room(room, "current_assignable")

        if "historical_room_context_not_assignable_count" in payload:
            add(
                "kg",
                "historical_room_context_not_assignable_count",
                payload[
                    "historical_room_context_not_assignable_count"
                ],
            )
        for room in payload.get(
                "historical_room_context_not_assignable", []):
            add_room(room, "historical_not_assignable")

        for room in payload.get("unexplored_context", []):
            add_room(room, "unexplored_context")

        if "explored_room_count" in payload:
            add("kg", "explored_room_count", payload["explored_room_count"])
        for room in payload.get("explored_rooms", []):
            add_room(room, "explored")

        return json.dumps(
            triples,
            ensure_ascii=False,
            separators=(",", ":"),
        )


# ═══════════════════════════════════════
# KG UPDATER
# ═══════════════════════════════════════

MERGE_DISTANCE = 30  # pixels — same object if within this distance
DYNAMIC_RELATIONS = {
    "in",
    "explored",
    "path_to",
    "next_to",
    "near",
    "connected_to",
    "separated_by_wall",
}
TRANSIENT_ROOM_PROPERTIES = {
    "frontier_idx",
    "frontier_indices",
    "size",
    "active_frontier",
    "target_prior",
}


class KGUpdater:
    def __init__(self, kg: KnowledgeGraph,
                 frame: Optional[EpisodeFrame] = None,
                 persistent_scene: bool = False,
                 episode_id: Optional[str] = None):
        self.kg = kg
        self._grid_size = 50
        self._map_size = MAP_SIZE
        self._update_id = 0
        self.frame = frame
        self.persistent_scene = bool(persistent_scene)
        self.episode_id = None if episode_id is None else str(episode_id)

    @property
    def _world_mode(self):
        return self.frame is not None

    def _episode_properties(self, **properties):
        if self.episode_id is not None:
            properties["episode_ids"] = [self.episode_id]
        return properties

    def _map_to_graph_position(self, row, col):
        if self.frame is None:
            return float(row), float(col)
        return self.frame.map_rc_to_world_xz(row, col)

    def _graph_position_to_room_id(self, position):
        if self.frame is None:
            return self._pos_to_room_id(*position)
        world_x, world_z = position
        grid_x = math.floor(float(world_x) / ROOM_CELL_M)
        grid_z = math.floor(float(world_z) / ROOM_CELL_M)
        return (
            f"scene_room_f{self.frame.floor_id}_x{grid_x}_z{grid_z}"
        )

    def _pos_to_room_id(self, y, x):
        if self.frame is not None:
            return self.frame.room_id_from_map_rc(y, x)
        gy = int(y // self._grid_size)
        gx = int(x // self._grid_size)
        return f"room_{gy}_{gx}"

    def _room_center(self, room_id):
        if room_id.startswith("scene_room_f"):
            try:
                _, _, floor, x_token, z_token = room_id.split("_")
                del floor
                grid_x = int(x_token[1:])
                grid_z = int(z_token[1:])
                return (
                    (grid_x + 0.5) * ROOM_CELL_M,
                    (grid_z + 0.5) * ROOM_CELL_M,
                )
            except (TypeError, ValueError):
                return (0, 0)
        try:
            _, gy, gx = room_id.split("_")
            return (
                (int(gy) + 0.5) * self._grid_size,
                (int(gx) + 0.5) * self._grid_size,
            )
        except (TypeError, ValueError):
            return (0, 0)

    def build_current_frontier_view(self, enriched_frontiers):
        """Build the per-decision mapping without mutating the persistent KG."""
        world_position_resolver = None
        if self.frame is not None:
            def world_position_resolver(row, col):
                return (
                    self.frame.map_rc_to_world_xz(row, col),
                    self.frame.floor_id,
                )
        return CurrentFrontierView.from_enriched(
            enriched_frontiers,
            room_id_resolver=self._pos_to_room_id,
            world_position_resolver=world_position_resolver,
        )

    def _update_room(self, room_id, categories, room_hint,
                     room_hint_confidence, explored, position=None,
                     local_position=None):
        existing = self.kg.nodes.get(room_id)
        categories = sorted(set(str(category) for category in categories))
        clean_hint = str(room_hint).replace("likely_", "") or "unknown"
        hint_confidence = float(np.clip(
            room_hint_confidence, 0.0, 1.0
        ))
        signature = {
            "categories": categories,
            "room_hint": clean_hint,
            "room_hint_confidence": round(hint_confidence, 2),
        }
        signatures = list(
            existing.properties.get("room_evidence_signatures", [])
            if existing is not None else []
        )
        is_new_evidence = signature not in signatures
        if is_new_evidence:
            signatures.append(signature)

        previous_hint_confidence = float(
            existing.properties.get("room_hint_confidence", 0.0)
            if existing is not None else 0.0
        )
        if (clean_hint not in ("", "unknown")
                and hint_confidence >= previous_hint_confidence):
            stored_name = clean_hint
            stored_hint_confidence = hint_confidence
        elif existing is not None:
            stored_name = existing.name or "unknown"
            stored_hint_confidence = previous_hint_confidence
        else:
            stored_name = "unknown"
            stored_hint_confidence = 0.0

        observed_categories = set(
            existing.properties.get("observed_categories", [])
            if existing is not None else []
        )
        observed_categories.update(categories)
        was_explored = bool(
            existing and existing.properties.get("explored", False)
        )
        previous_count = int(
            existing.properties.get("observation_count", 0)
            if existing is not None else 0
        )
        observation_count = previous_count + int(is_new_evidence)
        if observation_count == 0:
            observation_count = 1
        # Scene-room nodes represent fixed 2.5 m cells.  Keep their graph
        # position at the cell center so cross-episode topology is not
        # perturbed by whichever frontier happened to observe the cell last.
        graph_position = (
            self._room_center(room_id) if self._world_mode
            else (position if position is not None else self._room_center(room_id))
        )
        node = KGNode(
            id=room_id,
            node_type="room",
            name=stored_name,
            certainty=(
                stored_hint_confidence
                if stored_name != "unknown" else 0.1
            ),
            position=graph_position,
            properties=self._episode_properties(
                coordinate_frame=("world" if self._world_mode else "local"),
                floor_id=(self.frame.floor_id if self.frame else None),
                position_rc=(
                    [float(local_position[0]), float(local_position[1])]
                    if local_position is not None else None
                ),
                position_world_xz=(
                    [float(graph_position[0]), float(graph_position[1])]
                    if self._world_mode else None
                ),
                explored=bool(explored or was_explored),
                observation_count=observation_count,
                last_seen_update=self._update_id,
                room_hint_confidence=stored_hint_confidence,
                observed_categories=sorted(observed_categories),
                room_evidence_signatures=signatures[-32:],
            ),
        )
        self.kg.add_node(node)
        stored = self.kg.nodes[room_id]
        stored.name = node.name
        stored.certainty = node.certainty
        stored.position = node.position
        stored.properties.update(node.properties)

    def _find_or_create_object(self, category, pos, confidence,
                               observation_signature=None,
                               local_position=None):
        """Merge one unique map observation with paper-defined noisy-OR."""
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if observation_signature is None:
            observation_signature = (
                f"{category}:"
                f"{int(round(pos[0] / 4.0))}:"
                f"{int(round(pos[1] / 4.0))}"
            )
        candidates = []
        for node in self.kg.get_nodes_by_type("object"):
            if node.properties.get("category") == category:
                distance = float(np.hypot(
                    node.position[0] - pos[0],
                    node.position[1] - pos[1],
                ))
                merge_distance = 1.5 if self._world_mode else MERGE_DISTANCE
                same_floor = (
                    not self._world_mode
                    or node.properties.get("floor_id") == self.frame.floor_id
                )
                if distance < merge_distance and same_floor:
                    candidates.append((distance, node))

        if candidates:
            _, node = min(candidates, key=lambda item: item[0])
            signatures = list(
                node.properties.get("observation_signatures", [])
            )
            if observation_signature not in signatures:
                # Equation (1) in the paper.  De-duplicating cumulative map
                # contours first prevents the same snapshot from being counted
                # as an independent detection every 25 steps.
                node.certainty = min(
                    1.0 - (1.0 - float(node.certainty))
                    * (1.0 - confidence),
                    0.99,
                )
                node.position = (
                    0.7 * node.position[0] + 0.3 * pos[0],
                    0.7 * node.position[1] + 0.3 * pos[1],
                )
                if local_position is not None:
                    node.properties["position_rc"] = [
                        float(local_position[0]), float(local_position[1])
                    ]
                if self._world_mode:
                    node.properties["position_world_xz"] = [
                        float(node.position[0]), float(node.position[1])
                    ]
                count = node.properties.get("observation_count", 1) + 1
                node.properties["observation_count"] = count
                node.properties["detection_count"] = count
                signatures.append(observation_signature)
                node.properties["observation_signatures"] = signatures[-64:]
                detections = list(node.properties.get("detections", []))
                detections.append({
                    "confidence": confidence,
                    "position_rc": (
                        [float(local_position[0]), float(local_position[1])]
                        if local_position is not None else None
                    ),
                    "position_world_xz": (
                        [float(pos[0]), float(pos[1])]
                        if self._world_mode else None
                    ),
                })
                node.properties["detections"] = detections[-64:]
            node.properties["last_seen_update"] = self._update_id
            if self.episode_id is not None:
                episode_ids = set(node.properties.get("episode_ids", []))
                episode_ids.add(self.episode_id)
                node.properties["episode_ids"] = sorted(episode_ids)
                node.properties["episode_support_count"] = len(episode_ids)
            return node.id, False

        # Create new object node
        # ``pos`` is already in the graph frame. In scene mode this is a
        # Habitat world ``(x, z)`` position, so transforming it as map pixels
        # a second time would create an unstable, incorrect object ID.
        room_id = self._graph_position_to_room_id(pos)
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
            properties=self._episode_properties(
                coordinate_frame=("world" if self._world_mode else "local"),
                floor_id=(self.frame.floor_id if self.frame else None),
                position_rc=(
                    [float(local_position[0]), float(local_position[1])]
                    if local_position is not None else [float(pos[0]), float(pos[1])]
                ),
                position_world_xz=(
                    [float(pos[0]), float(pos[1])] if self._world_mode else None
                ),
                category=category,
                observation_count=1,
                detection_count=1,
                first_seen_update=self._update_id,
                last_seen_update=self._update_id,
                observation_signatures=[observation_signature],
                detections=[{
                    "confidence": confidence,
                    "position_rc": (
                        [float(local_position[0]), float(local_position[1])]
                        if local_position is not None else [float(pos[0]), float(pos[1])]
                    ),
                    "position_world_xz": (
                        [float(pos[0]), float(pos[1])]
                        if self._world_mode else None
                    ),
                }],
            )
        ))
        if self.episode_id is not None:
            self.kg.nodes[obj_id].properties["episode_support_count"] = 1
        return obj_id, True

    @staticmethod
    def _object_component_position(component):
        """Return map ``(row, col)`` from an OpenCV ``(x, y)`` contour."""
        points = np.asarray(component, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return None
        col = float(points[:, 0].mean())
        row = float(points[:, 1].mean())
        if not np.isfinite(row) or not np.isfinite(col):
            return None
        return row, col

    @staticmethod
    def _object_component_signature(category, component):
        """Stable signature for de-duplicating a cumulative map contour."""
        points = np.asarray(component, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return f"{category}:empty"
        values = (
            points[:, 0].mean(),
            points[:, 1].mean(),
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max(),
        )
        quantized = [int(round(value / 4.0)) for value in values]
        return f"{category}:" + ":".join(map(str, quantized))

    @staticmethod
    def _segments_intersect(a, b, c, d):
        def orientation(p, q, r):
            value = ((q[1] - p[1]) * (r[0] - q[0])
                     - (q[0] - p[0]) * (r[1] - q[1]))
            if abs(value) < 1e-6:
                return 0
            return 1 if value > 0 else 2

        return (
            orientation(a, b, c) != orientation(a, b, d)
            and orientation(c, d, a) != orientation(c, d, b)
        )

    def _wall_segments(self, wall_list):
        segments = []
        if wall_list is None:
            return segments
        for wall in wall_list:
            try:
                x1, y1, x2, y2 = np.asarray(wall).reshape(-1)[:4]
                segment = (
                    self._map_to_graph_position(float(y1), float(x1)),
                    self._map_to_graph_position(float(y2), float(x2)),
                )
                if all(np.isfinite(value) for point in segment for value in point):
                    segments.append(segment)
            except (TypeError, ValueError, IndexError):
                continue
        return segments

    def update(self, enriched_frontiers, object_list, pose_pred, wall_list):
        """Update persistent evidence and rebuild decision-time relations."""
        if not self.kg.nodes:
            self._update_id = 0
        self._update_id += 1

        for room in self.kg.get_nodes_by_type("room"):
            for property_name in TRANSIENT_ROOM_PROPERTIES:
                room.properties.pop(property_name, None)

        dynamic_relations = (
            {"in", "explored", "path_to"}
            if self.persistent_scene else DYNAMIC_RELATIONS
        )
        self.kg.remove_edges(lambda edge: edge.relation in dynamic_relations)

        room_observations = {}

        def observe_room(room_id, position, categories=(),
                         room_hint="unknown", room_hint_confidence=0.0,
                         explored=False, local_position=None):
            observation = room_observations.setdefault(room_id, {
                "categories": set(),
                "room_hint": "unknown",
                "room_hint_confidence": 0.0,
                "explored": False,
                "position": position,
                "local_position": local_position,
            })
            observation["categories"].update(categories)
            if float(room_hint_confidence) >= observation["room_hint_confidence"]:
                observation["room_hint"] = room_hint
                observation["room_hint_confidence"] = float(
                    room_hint_confidence
                )
            observation["explored"] = bool(
                observation["explored"] or explored
            )

        for frontier in enriched_frontiers:
            row, col = map(float, frontier["centroid"][:2])
            room_id = self._pos_to_room_id(row, col)
            observe_room(
                room_id,
                self._map_to_graph_position(row, col),
                categories=frontier.get("nearby_objects", ()),
                room_hint=frontier.get("room_type", "unknown"),
                room_hint_confidence=frontier.get("room_confidence", 0.0),
                explored=bool(
                    self.kg.nodes.get(room_id)
                    and self.kg.nodes[room_id].properties.get("explored")
                ),
                local_position=(row, col),
            )

        object_room_links = {}
        if object_list:
            for object_name, components in object_list.items():
                for component in components[:5]:
                    object_position_rc = self._object_component_position(component)
                    if (object_position_rc is None
                            or object_position_rc == (0, 0)):
                        continue
                    object_position = self._map_to_graph_position(
                        *object_position_rc
                    )
                    signature = self._object_component_signature(
                        object_name, component
                    )
                    if self._world_mode:
                        signature = (
                            f"{object_name}:world:"
                            f"{int(round(object_position[0] / 0.2))}:"
                            f"{int(round(object_position[1] / 0.2))}"
                        )
                    object_id, _ = self._find_or_create_object(
                        object_name,
                        object_position,
                        confidence=0.65,
                        observation_signature=signature,
                        local_position=object_position_rc,
                    )
                    stored_position = self.kg.nodes[object_id].position
                    room_id = self._graph_position_to_room_id(stored_position)
                    object_room_links[object_id] = room_id
                    observe_room(
                        room_id,
                        stored_position,
                        categories=(object_name,),
                        local_position=object_position_rc,
                    )

        robot_records = []
        for robot_index, pose_value in enumerate(pose_pred):
            map_position = CurrentFrontierView.robot_pose_to_map_rc(
                pose_value,
                self._map_size,
            )
            if map_position is None:
                continue
            robot_id = f"robot_{robot_index}"
            graph_position = self._map_to_graph_position(*map_position)
            room_id = self._graph_position_to_room_id(graph_position)
            robot_records.append(
                (robot_id, room_id, graph_position, map_position)
            )
            observe_room(
                room_id, graph_position, explored=True,
                local_position=map_position,
            )

        for room_id, observation in room_observations.items():
            self._update_room(
                room_id,
                observation["categories"],
                observation["room_hint"],
                observation["room_hint_confidence"],
                observation["explored"],
                position=observation["position"],
                local_position=observation["local_position"],
            )

        for object_id, room_id in object_room_links.items():
            self.kg.remove_edges(
                lambda edge, target=object_id: (
                    edge.relation == "contains" and edge.target == target
                )
            )
            self.kg.add_edge(KGEdge(
                room_id,
                object_id,
                "contains",
                properties=self._episode_properties(
                    last_seen_update=self._update_id,
                ),
            ))

        for robot_id, room_id, graph_position, map_position in robot_records:
            self.kg.add_node(KGNode(
                id=robot_id,
                node_type="robot",
                name=robot_id,
                certainty=1.0,
                position=graph_position,
                properties=self._episode_properties(
                    coordinate_frame=("world" if self._world_mode else "local"),
                    floor_id=(self.frame.floor_id if self.frame else None),
                    position_rc=[
                        float(map_position[0]), float(map_position[1])
                    ],
                    position_world_xz=(
                        [float(graph_position[0]), float(graph_position[1])]
                        if self._world_mode else None
                    ),
                    last_seen_update=self._update_id,
                ),
            ))
            self.kg.add_edge(KGEdge(robot_id, room_id, "in"))
            self.kg.add_edge(KGEdge(robot_id, room_id, "explored"))

        objects = self.kg.get_nodes_by_type("object")
        object_next_to_threshold = 1.0 if self._world_mode else 20
        object_near_threshold = 3.0 if self._world_mode else 60
        for first_index, first in enumerate(objects):
            for second in objects[first_index + 1:]:
                if (self._world_mode
                        and first.properties.get("floor_id")
                        != second.properties.get("floor_id")):
                    continue
                if first.position == (0, 0) or second.position == (0, 0):
                    continue
                distance = float(np.hypot(
                    first.position[0] - second.position[0],
                    first.position[1] - second.position[1],
                ))
                if distance < object_next_to_threshold:
                    self.kg.add_edge(KGEdge(
                        first.id, second.id, "next_to", distance=distance,
                        properties=self._episode_properties(
                            last_seen_update=self._update_id,
                        ),
                    ))
                elif distance < object_near_threshold:
                    self.kg.add_edge(KGEdge(
                        first.id, second.id, "near", distance=distance,
                        properties=self._episode_properties(
                            last_seen_update=self._update_id,
                        ),
                    ))

        wall_segments = self._wall_segments(wall_list)
        rooms = self.kg.get_nodes_by_type("room")
        current_rooms = [
            self.kg.nodes[room_id] for room_id in room_observations
            if room_id in self.kg.nodes
        ]
        connection_threshold = 3.75 if self._world_mode else 150
        for first_index, first in enumerate(current_rooms):
            for second in current_rooms[first_index + 1:]:
                if (self._world_mode
                        and first.properties.get("floor_id")
                        != second.properties.get("floor_id")):
                    continue
                distance = float(np.hypot(
                    first.position[0] - second.position[0],
                    first.position[1] - second.position[1],
                ))
                if distance >= connection_threshold:
                    continue
                blocked = any(
                    self._segments_intersect(
                        first.position,
                        second.position,
                        wall_start,
                        wall_end,
                    )
                    for wall_start, wall_end in wall_segments
                )
                relation = "separated_by_wall" if blocked else "connected_to"
                confidence = 0.75 if blocked else 0.40
                self.kg.add_edge(KGEdge(
                    first.id,
                    second.id,
                    relation,
                    distance=distance,
                    properties=self._episode_properties(
                        confidence=confidence,
                        last_seen_update=self._update_id,
                    ),
                ))

        # Compute reachability only after all rooms for this update exist.
        reachability_threshold = 15.0 if self._world_mode else 300
        for robot_id, _, graph_position, _ in robot_records:
            for room in rooms:
                if room.properties.get("explored"):
                    continue
                distance = float(np.hypot(
                    graph_position[0] - room.position[0],
                    graph_position[1] - room.position[1],
                ))
                if distance < reachability_threshold:
                    self.kg.add_edge(KGEdge(
                        robot_id,
                        room.id,
                        "path_to",
                        distance=distance,
                        properties=self._episode_properties(
                            last_seen_update=self._update_id,
                        ),
                    ))
