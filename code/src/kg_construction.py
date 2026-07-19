"""MindNav persistent evidence graph and current-frontier view.

The graph keeps episode-level region/object evidence, while executable frontier
indices live only in :class:`CurrentFrontierView`.  All stored geometry uses map
``(row, column)`` coordinates.
"""
import json

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


MAP_SIZE = 480
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
class CurrentFrontier:
    """One executable frontier candidate for the current planning step only."""

    idx: int
    room_id: str
    centroid: Tuple[float, float]
    area: float
    target_prior: float = 0.0
    room_type: str = "unknown"
    room_confidence: float = 0.0


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
                      grid_size=50):
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
            raw_position = (
                pose_pred[robot_index]
                if robot_index < len(pose_pred) else None
            )
            position = self.robot_pose_to_map_rc(raw_position)
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
            existing.properties.update(node.properties)
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
                    e.properties.update(edge.properties)
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

    def to_text(self, current_frontier_view=None,
                max_historical_rooms=6) -> str:
        """Render persistent context and current assignable rooms separately."""
        lines = []
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
            path_str = ", ".join([f"{p[0]}({p[1]:.0f}px)" for p in paths[:5]])
            lines.append(
                f"{r.id}: position_rc=({r.position[0]:.0f},"
                f"{r.position[1]:.0f}), in={in_room}, "
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
                    connections.append(f"{other}({edge.distance:.0f}px)")
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
                                f"({object_edge.distance:.0f}px)──> {other.name}"
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

        def rounded(value):
            return int(round(float(value)))

        def percent(value):
            return int(round(100.0 * float(value)))

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
            robots.append({
                "id": robot.id,
                "position_rc": [
                    rounded(robot.position[0]),
                    rounded(robot.position[1]),
                ],
                "in_room": in_room,
                "explored_room_count": len(explored),
                path_label: [
                    {"room_id": room_id, "distance_px": rounded(distance)}
                    for room_id, distance in paths[:5]
                ],
            })

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
                        "distance_px": rounded(edge.distance),
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
                                "distance_px": rounded(edge.distance),
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
            "format": "mindnav_kg_json_v1",
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
            "mindnav_kg_triples_v1",
        ]]

        def add(head, relation, tail):
            triples.append([head, relation, tail])

        for robot_index, robot in enumerate(payload.get("robots", [])):
            robot_id = robot["id"]
            add(robot_id, "node_type", "robot")
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
                add(path_ref, "distance_px", path["distance_px"])

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
                    "distance_px",
                    relation["distance_px"],
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
                    "distance_px",
                    connection["distance_px"],
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
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._grid_size = 50
        self._map_size = MAP_SIZE
        self._update_id = 0

    def _pos_to_room_id(self, y, x):
        gy = int(y // self._grid_size)
        gx = int(x // self._grid_size)
        return f"room_{gy}_{gx}"

    def _room_center(self, room_id):
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
        return CurrentFrontierView.from_enriched(
            enriched_frontiers,
            room_id_resolver=self._pos_to_room_id,
        )

    def _update_room(self, room_id, categories, room_hint,
                     room_hint_confidence, explored):
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
        node = KGNode(
            id=room_id,
            node_type="room",
            name=stored_name,
            certainty=(
                stored_hint_confidence
                if stored_name != "unknown" else 0.1
            ),
            position=self._room_center(room_id),
            properties={
                "explored": bool(explored or was_explored),
                "observation_count": observation_count,
                "last_seen_update": self._update_id,
                "room_hint_confidence": stored_hint_confidence,
                "observed_categories": sorted(observed_categories),
                "room_evidence_signatures": signatures[-32:],
            },
        )
        self.kg.add_node(node)
        stored = self.kg.nodes[room_id]
        stored.name = node.name
        stored.certainty = node.certainty
        stored.position = self._room_center(room_id)
        stored.properties.update(node.properties)

    def _find_or_create_object(self, category, pos, confidence,
                               observation_signature=None):
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
                if distance < MERGE_DISTANCE:
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
                count = node.properties.get("observation_count", 1) + 1
                node.properties["observation_count"] = count
                node.properties["detection_count"] = count
                signatures.append(observation_signature)
                node.properties["observation_signatures"] = signatures[-64:]
                detections = list(node.properties.get("detections", []))
                detections.append({
                    "confidence": confidence,
                    "position_rc": [float(pos[0]), float(pos[1])],
                })
                node.properties["detections"] = detections[-64:]
            node.properties["last_seen_update"] = self._update_id
            return node.id, False

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
            properties={
                "category": category,
                "observation_count": 1,
                "detection_count": 1,
                "first_seen_update": self._update_id,
                "last_seen_update": self._update_id,
                "observation_signatures": [observation_signature],
                "detections": [{
                    "confidence": confidence,
                    "position_rc": [float(pos[0]), float(pos[1])],
                }],
            }
        ))
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

    @staticmethod
    def _wall_segments(wall_list):
        segments = []
        if wall_list is None:
            return segments
        for wall in wall_list:
            try:
                x1, y1, x2, y2 = np.asarray(wall).reshape(-1)[:4]
                segment = ((float(y1), float(x1)),
                           (float(y2), float(x2)))
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

        self.kg.remove_edges(
            lambda edge: edge.relation in DYNAMIC_RELATIONS
        )

        room_observations = {}

        def observe_room(room_id, categories=(), room_hint="unknown",
                         room_hint_confidence=0.0, explored=False):
            observation = room_observations.setdefault(room_id, {
                "categories": set(),
                "room_hint": "unknown",
                "room_hint_confidence": 0.0,
                "explored": False,
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
                categories=frontier.get("nearby_objects", ()),
                room_hint=frontier.get("room_type", "unknown"),
                room_hint_confidence=frontier.get("room_confidence", 0.0),
                explored=bool(
                    self.kg.nodes.get(room_id)
                    and self.kg.nodes[room_id].properties.get("explored")
                ),
            )

        object_room_links = {}
        if object_list:
            for object_name, components in object_list.items():
                for component in components[:5]:
                    object_position = self._object_component_position(component)
                    if object_position is None or object_position == (0, 0):
                        continue
                    object_id, _ = self._find_or_create_object(
                        object_name,
                        object_position,
                        confidence=0.65,
                        observation_signature=(
                            self._object_component_signature(
                                object_name, component
                            )
                        ),
                    )
                    stored_position = self.kg.nodes[object_id].position
                    room_id = self._pos_to_room_id(*stored_position)
                    object_room_links[object_id] = room_id
                    observe_room(room_id, categories=(object_name,))

        robot_records = []
        for robot_index, raw_pose in enumerate(pose_pred):
            map_position = CurrentFrontierView.robot_pose_to_map_rc(
                raw_pose,
                self._map_size,
            )
            if map_position is None:
                continue
            robot_id = f"robot_{robot_index}"
            room_id = self._pos_to_room_id(*map_position)
            robot_records.append((robot_id, room_id, map_position))
            observe_room(room_id, explored=True)

        for room_id, observation in room_observations.items():
            self._update_room(
                room_id,
                observation["categories"],
                observation["room_hint"],
                observation["room_hint_confidence"],
                observation["explored"],
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
                properties={"last_seen_update": self._update_id},
            ))

        for robot_id, room_id, map_position in robot_records:
            self.kg.add_node(KGNode(
                id=robot_id,
                node_type="robot",
                name=robot_id,
                certainty=1.0,
                position=map_position,
                properties={"last_seen_update": self._update_id},
            ))
            self.kg.add_edge(KGEdge(robot_id, room_id, "in"))
            self.kg.add_edge(KGEdge(robot_id, room_id, "explored"))

        objects = self.kg.get_nodes_by_type("object")
        for first_index, first in enumerate(objects):
            for second in objects[first_index + 1:]:
                if first.position == (0, 0) or second.position == (0, 0):
                    continue
                distance = float(np.hypot(
                    first.position[0] - second.position[0],
                    first.position[1] - second.position[1],
                ))
                if distance < 20:
                    self.kg.add_edge(KGEdge(
                        first.id, second.id, "next_to", distance=distance
                    ))
                elif distance < 60:
                    self.kg.add_edge(KGEdge(
                        first.id, second.id, "near", distance=distance
                    ))

        wall_segments = self._wall_segments(wall_list)
        rooms = self.kg.get_nodes_by_type("room")
        for first_index, first in enumerate(rooms):
            for second in rooms[first_index + 1:]:
                distance = float(np.hypot(
                    first.position[0] - second.position[0],
                    first.position[1] - second.position[1],
                ))
                if distance >= 150:
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
                    properties={
                        "confidence": confidence,
                        "last_seen_update": self._update_id,
                    },
                ))

        # Compute reachability only after all rooms for this update exist.
        for robot_id, _, map_position in robot_records:
            for room in rooms:
                if room.properties.get("explored"):
                    continue
                distance = float(np.hypot(
                    map_position[0] - room.position[0],
                    map_position[1] - room.position[1],
                ))
                if distance < 300:
                    self.kg.add_edge(KGEdge(
                        robot_id,
                        room.id,
                        "path_to",
                        distance=distance,
                        properties={"last_seen_update": self._update_id},
                    ))
