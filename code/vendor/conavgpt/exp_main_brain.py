from collections import deque, defaultdict
import copy
from typing import Dict
from itertools import count
import os
import sys
import logging
import time
import json
import re
import subprocess
from datetime import datetime, timezone
import gym
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from agents.llm_agents import LLM_Agent
from envs.habitat.multi_agent_env import Multi_Agent_Env
from constants import color_palette, coco_categories, hm3d_category, category_to_id
import utils.visualization as vu
from arguments import get_args

# MindNav core lives in code/src. Keep it canonical instead of importing the
# duplicated agents/helicase*.py copies under the vendored runtime.
CODE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if CODE_SRC not in sys.path:
    sys.path.insert(0, CODE_SRC)
from brain import HelicaseBrain
from kg_construction import (
    MAP_SIZE, CurrentFrontierView, EpisodeFrame, KnowledgeGraph, KGUpdater,
)
from llm_api import OpenAICompatibleBrainAdapter, load_brain_api_config
from reproducibility import reset_episode_rng

from skimage import measure
import skimage.morphology
from PIL import Image

import cv2
import habitat
# Local VLM replaces OpenAI
from local_vlm import load_model, chat_completion_create

import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

system_prompt = """Objective: Control two robots equipped with cameras and 5-meter range depth sensors. They must collaboratively explore an indoor home scene on a 2D map to locate a specific target object.

Map Information:
Origin: Top-left corner.
Coordinates: Represented in pixels.

Representation Details:
Robot Position: Format - (x, y).
Scene Objects: Defined by vertices in clockwise order. Format - <(x1, y1), (x2, y2)...>.
Walls: Lines with start and end points. Format - <(x1_start, y1_start, x1_end, y1_end)>.
Frontiers: Defined by centroid, pixel count, inferred room type, nearby objects, and target likelihood. Format - <centroid:(x, y), number: n, room_type: type, nearby: obj1, obj2, target_likelihood: p>.
The target_likelihood indicates how likely the target object is to be found through this frontier (higher = better).

Strategy Considerations:
CRITICAL RULE: Each robot MUST explore a DIFFERENT frontier. NEVER assign both robots to the same frontier — this wastes exploration effort.
Prioritize frontiers with HIGH target_likelihood — these lead toward rooms where the target is most likely found.
Use room_type to reason: toilets are in bathrooms, beds are in bedrooms, TVs/sofas/chairs are in living rooms, plants are in hallways or living rooms.
Do NOT default to the largest frontier — choose based on target_likelihood and room_type, not frontier size.
Also consider frontier proximity and previous movement directions.
Minimize frequent switches unless a frontier with significantly higher target_likelihood is discovered.

Instructions: Given the scene details, decide the best frontier each robot should explore next. Provide ONLY the decision in the [output:] format without additional explanations or additional text.

Example: 

[input:]
Task: locate the chairs

Position: 
robot_0: (240, 240)
robot_1: (200, 150) 

Scene Objects:
sofa: <(280, 200), (280, 150), (330, 150), (330, 180), (300, 180), (300, 200)> 
bed: <(220, 250), (240, 250), (250, 250), (250, 220)> 
chest_of_drawers: <(200, 240), (210, 240), (210, 250), (200, 250)> 
tv_monitor: <(220, 200), (240, 200), (240, 210), (220, 210)> 
table: <(200, 150), (230, 150), (230, 170), (200, 170)> 

Walls:
wall_0: <(190, 180, 300, 180)> 
wall_1: <(160, 180, 160, 250)> 
wall_2: <(160, 250, 30, 250)> 

Previous Movements:
robot_0: <centroid:(195, 320), number: 60>
robot_1: <centroid:(170, 180), number: 30> 

Unexplored Frontier:
frontier_0: <centroid:(180, 175), number: 40, room_type: kitchen, nearby: table, target_likelihood: 0.3>
frontier_1: <centroid:(195, 280), number: 80, room_type: living_room, nearby: sofa, tv_monitor, target_likelihood: 0.5>

[output:]
robot_0: frontier_1
robot_1: frontier_0 

Please give the output based on the following input:\n"""

# Local LLM model paths (text-only, matching Co-NavGPT's text-only usage)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
LOCAL_MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, 'models', 'Qwen2.5-3B-Instruct'),
    os.path.join(PROJECT_ROOT, 'models', 'Qwen2.5-7B-Instruct'),
]
gpt_name = ['Qwen2.5-3B', 'Qwen2.5-7B']


def build_room_frontier_audit(
        episode_index, episode_id, step, goal_name, enriched_frontiers,
        current_frontiers, kg, llm_response, final_goal_frontiers,
        decision_audit, decision_method, tools_called):
    """Audit that every executed frontier belongs to the current view."""
    active_frontiers = []
    for frontier_idx in sorted(current_frontiers.frontiers):
        frontier = current_frontiers.frontiers[frontier_idx]
        active_frontiers.append({
            "frontier_idx": frontier.idx,
            "room_id": frontier.room_id,
            "centroid": list(frontier.centroid),
            "area": int(frontier.area),
            "room_type": frontier.room_type,
            "room_confidence": frontier.room_confidence,
            "python_prior": frontier.target_prior,
        })

    group_records = []
    for room_id, frontier_indices in sorted(
            current_frontiers.room_to_frontiers.items()):
        group_records.append({
            "room_id": room_id,
            "active_frontier_indices": list(frontier_indices),
            "collision": len(frontier_indices) > 1,
            "all_frontiers_preserved": all(
                frontier_idx in current_frontiers.frontiers
                for frontier_idx in frontier_indices
            ),
        })

    current_room_ids = current_frontiers.room_ids
    allowed_room_ids = set(decision_audit.get("allowed_room_ids", []))
    exposed_noncurrent_rooms = sorted(allowed_room_ids - current_room_ids)
    omitted_current_rooms = sorted(current_room_ids - allowed_room_ids)

    transient_properties = {
        "frontier_idx", "frontier_indices", "size",
        "active_frontier", "target_prior",
    }
    persistent_transient_rooms = []
    for node in kg.get_nodes_by_type("room"):
        leaked = sorted(transient_properties & set(node.properties))
        if leaked:
            persistent_transient_rooms.append({
                "room_id": node.id,
                "properties": leaked,
            })

    kg_candidates = []
    for room_id in sorted(current_room_ids):
        node = kg.nodes.get(room_id)
        kg_candidates.append({
            "room_id": room_id,
            "active_frontier_indices": list(
                current_frontiers.room_to_frontiers[room_id]
            ),
            "position": (
                [float(node.position[0]), float(node.position[1])]
                if node is not None else None
            ),
            "is_current_room": True,
        })

    llm_probabilities = {}
    for match in re.finditer(
            r'P\([^|]+\|(room_\d+_\d+)\)\s*=\s*([\d.]+)', llm_response):
        llm_probabilities[match.group(1)] = float(match.group(2))

    llm_room_assignments = {}
    for match in re.finditer(
            r'robot_(\d+)\s*:\s*(room_\d+_\d+)', llm_response):
        llm_room_assignments[f"robot_{match.group(1)}"] = match.group(2)

    scored_room_ids = set(llm_probabilities)
    assigned_room_ids = set(llm_room_assignments.values())
    selected_rooms = decision_audit.get("selected_room_by_robot", {})
    accepted_rooms = decision_audit.get("accepted_llm_rooms", {})

    assignment_mapping = {}
    executed_noncurrent = []
    room_frontier_mismatches = []
    robot_ids = sorted(set(final_goal_frontiers) | set(llm_room_assignments))
    for robot_id in robot_ids:
        frontier_idx = int(final_goal_frontiers.get(robot_id, -1))
        frontier = current_frontiers.frontiers.get(frontier_idx)
        frontier_room_id = frontier.room_id if frontier is not None else None
        selected_room_id = selected_rooms.get(robot_id)
        valid_current = frontier is not None
        selected_matches = bool(
            valid_current and selected_room_id == frontier_room_id
        )
        assignment_mapping[robot_id] = {
            "llm_room_id": llm_room_assignments.get(robot_id),
            "llm_room_is_current": (
                llm_room_assignments.get(robot_id) in current_room_ids
            ),
            "accepted_llm_room_id": accepted_rooms.get(robot_id),
            "selected_room_id": selected_room_id,
            "final_frontier_idx": frontier_idx,
            "frontier_room_id": frontier_room_id,
            "valid_current_frontier": valid_current,
            "selected_room_matches_frontier": selected_matches,
        }
        if not valid_current:
            executed_noncurrent.append(robot_id)
        if valid_current and not selected_matches:
            room_frontier_mismatches.append(robot_id)

    final_indices = [
        int(frontier_idx) for frontier_idx in final_goal_frontiers.values()
    ]
    duplicate_final_frontiers = (
        len(current_frontiers.frontiers) >= len(final_indices)
        and len(final_indices) != len(set(final_indices))
    )

    collision_groups = [entry for entry in group_records if entry["collision"]]
    rejected_room_ids = decision_audit.get("rejected_room_ids", [])

    return {
        "schema_version": 2,
        "episode_index": int(episode_index),
        "episode": int(episode_index) + 1,
        "episode_id": str(episode_id),
        "step": int(step),
        "goal": goal_name,
        "decision_method": decision_method,
        "tools": list(tools_called),
        "active_frontiers": active_frontiers,
        "current_room_groups": group_records,
        "kg_candidate_rooms": kg_candidates,
        "llm_probabilities": llm_probabilities,
        "llm_room_assignments": llm_room_assignments,
        "assignment_mapping": assignment_mapping,
        "decision_audit": dict(decision_audit),
        "final_frontier_assignments": {
            key: int(value) for key, value in final_goal_frontiers.items()
        },
        "issues": {
            "collision_room_ids": [entry["room_id"] for entry in collision_groups],
            "overwritten_active_frontier_count": 0,
            "stale_candidate_room_ids": exposed_noncurrent_rooms,
            "misbound_candidate_room_ids": [],
            "llm_scored_noncurrent_room_ids": sorted(scored_room_ids - current_room_ids),
            "llm_omitted_current_candidate_room_ids": sorted(current_room_ids - scored_room_ids),
            "llm_assigned_noncurrent_room_ids": sorted(
                assigned_room_ids - current_room_ids
            ),
            "rejected_llm_room_ids": list(rejected_room_ids),
            "omitted_current_room_ids_from_prompt": omitted_current_rooms,
            "persistent_transient_room_properties": persistent_transient_rooms,
            "executed_noncurrent_robot_ids": executed_noncurrent,
            "room_frontier_mismatch_robot_ids": room_frontier_mismatches,
            "duplicate_final_frontiers": duplicate_final_frontiers,
        },
        "counts": {
            "active_frontiers": len(active_frontiers),
            "unique_current_rooms": len(current_room_ids),
            "collision_rooms": len(collision_groups),
            "overwritten_active_frontiers": 0,
            "kg_candidate_rooms": len(current_room_ids),
            "stale_candidate_rooms": len(exposed_noncurrent_rooms),
            "misbound_candidate_rooms": 0,
            "llm_scored_rooms": len(scored_room_ids),
            "llm_scored_noncurrent_rooms": len(scored_room_ids - current_room_ids),
            "llm_omitted_current_candidate_rooms": len(current_room_ids - scored_room_ids),
            "llm_assigned_noncurrent_rooms": len(
                assigned_room_ids - current_room_ids
            ),
            "rejected_llm_rooms": len(rejected_room_ids),
            "persistent_transient_room_properties": len(
                persistent_transient_rooms
            ),
            "executed_noncurrent_frontiers": len(executed_noncurrent),
            "room_frontier_mismatches": len(room_frontier_mismatches),
            "duplicate_final_frontiers": int(duplicate_final_frontiers),
        },
    }


def build_kg_snapshot_record(
        episode_index, episode_id, scene, goal_name, step, kg,
        snapshot_type, metrics=None, frame=None, memory_mode="episode"):
    """Attach episode provenance to a complete persistent-KG snapshot."""
    if snapshot_type not in {"decision", "final"}:
        raise ValueError(f"Unsupported KG snapshot type: {snapshot_type}")
    record = {
        "schema_version": 2,
        "snapshot_type": snapshot_type,
        "episode_index": int(episode_index),
        "episode_id": str(episode_id),
        "scene": os.path.basename(scene),
        "goal": str(goal_name),
        "step": int(step),
        "kg_memory_mode": str(memory_mode),
    }
    if frame is not None:
        record.update({
            "coordinate_frame": "world",
            "start_position": [float(value) for value in frame.start_position],
            "start_rotation": [float(value) for value in frame.start_rotation],
            "floor_id": int(frame.floor_id),
            "map_size_px": int(frame.map_size_px),
            "map_resolution_m": float(frame.map_resolution_m),
        })
    record.update(kg.to_snapshot_dict())
    if metrics is not None:
        record["episode_metrics"] = {
            "habitat_success": float(metrics.get("success", 0.0)),
            "habitat_spl": float(metrics.get("spl", 0.0)),
            "distance_to_goal": float(
                metrics.get("distance_to_goal", -1.0)
            ),
        }
    return record


def get_per_agent_goal_distances(env, episode, num_agents):
    """Return oracle distances for diagnostics only, never for control."""
    try:
        view_points = [
            view_point.agent_state.position
            for goal in episode.goals
            for view_point in goal.view_points
        ]
        return [
            float(env.sim.geodesic_distance(
                env.sim.get_agent_state(agent_id=agent_id).position,
                view_points,
                episode,
            ))
            for agent_id in range(num_agents)
        ]
    except Exception as error:
        print(f"STOP diagnostic distance error: {error}")
        return [None] * num_agents


def save_target_diagnostic_artifact(
    agent,
    base_dir,
    episode_index,
    goal,
    step,
    event,
    action,
    global_distance_to_goal,
    agent_distance_to_goal,
):
    """Save passive RGB/mask/map evidence for a target event."""
    if agent.last_rgb_observation is None:
        return None

    event_dir = os.path.join(
        os.path.abspath(base_dir),
        f"episode_{episode_index + 1:02d}_{goal}",
    )
    os.makedirs(event_dir, exist_ok=True)
    stem = f"step_{step:03d}_robot_{agent.agent_id}_{event}"
    masks = agent.last_target_frame_masks
    rgb = agent.last_rgb_observation.astype(np.uint8, copy=True)
    rednet = masks.get("rednet", np.zeros(rgb.shape[:2], dtype=np.uint8))
    maskrcnn = masks.get(
        "maskrcnn", np.zeros(rgb.shape[:2], dtype=np.uint8)
    )
    fused = masks.get(
        "fused_union", np.zeros(rgb.shape[:2], dtype=np.uint8)
    )
    intersection = masks.get(
        "source_intersection", np.zeros(rgb.shape[:2], dtype=np.uint8)
    )

    panels = [rgb]
    for label, mask, color in (
        ("RedNet", rednet, (255, 0, 0)),
        ("Mask R-CNN", maskrcnn, (0, 0, 255)),
        ("Fused", fused, (255, 0, 255)),
    ):
        overlay = rgb.copy()
        overlay[mask.astype(bool)] = (
            0.45 * overlay[mask.astype(bool)]
            + 0.55 * np.asarray(color)
        ).astype(np.uint8)
        cv2.putText(
            overlay,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(overlay)
    composite = np.concatenate(panels, axis=1)
    image_path = os.path.join(event_dir, stem + ".png")
    cv2.imwrite(image_path, composite[:, :, ::-1])

    array_path = os.path.join(event_dir, stem + ".npz")
    np.savez_compressed(
        array_path,
        rgb=rgb,
        rednet=rednet,
        maskrcnn=maskrcnn,
        fused_union=fused,
        source_intersection=intersection,
        target_map=(
            agent.last_target_map_binary
            if agent.last_target_map_binary is not None
            else np.zeros((1, 1), dtype=np.uint8)
        ),
        goal_map=(
            agent.last_goal_map_binary
            if agent.last_goal_map_binary is not None
            else np.zeros((1, 1), dtype=np.uint8)
        ),
    )

    metadata = {
        "episode_index": int(episode_index),
        "episode": int(episode_index + 1),
        "goal": goal,
        "step": int(step),
        "robot_id": f"robot_{agent.agent_id}",
        "event": event,
        "action": int(action),
        "global_distance_to_goal": float(global_distance_to_goal),
        "agent_distance_to_goal": agent_distance_to_goal,
        "found_goal": int(agent.last_found_goal),
        "planner_stop": bool(agent.last_planner_stop),
        "target_map_mass": float(agent.last_target_map_mass),
        "goal_map_mass": float(agent.last_goal_map_mass),
        "semantic_sources": agent.last_semantic_source_stats,
        "target_map_stats": agent.last_target_map_stats,
        "image": image_path,
        "arrays": array_path,
    }
    metadata_path = os.path.join(event_dir, stem + ".json")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata_path


def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, l_step)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    no_cat_mask = sem_map == 20
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i


    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)

    vis_image[50:530, 15:495] = sem_map_vis

    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)

    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Vis-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        cv2.imwrite(fn, vis_image)

def Frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    # contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
    Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
            if i == 3:
                break
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Wall_lines, Goal_area_list, Goal_edge, Goal_point

# ═══════════════════════════════════════════════════════════════
# Semantic Frontier Enrichment (Iteration 1: room-type + nearby objects)
# ═══════════════════════════════════════════════════════════════

# Room type inference from nearby objects (expanded signatures for better coverage)
ROOM_SIGNATURES = {
    "bathroom": {"toilet", "bathtub", "shower", "sink", "towel"},
    "bedroom": {"bed", "chest_of_drawers"},
    "living_room": {"sofa", "tv_monitor", "fireplace", "chair"},
    "kitchen": {"appliances", "sink", "table"},
    "hallway": {"stairs"},
}

# Adjacency priors: if we've explored room X, what room is likely next?
# P(next_room | current_room) — from common house layouts
ROOM_ADJACENCY = {
    "bedroom": {"bathroom": 0.4, "hallway": 0.3, "bedroom": 0.2, "living_room": 0.1},
    "living_room": {"kitchen": 0.3, "hallway": 0.3, "bedroom": 0.2, "bathroom": 0.1, "living_room": 0.1},
    "kitchen": {"living_room": 0.3, "hallway": 0.2, "dining_room": 0.2, "bathroom": 0.1, "kitchen": 0.1},
    "hallway": {"bedroom": 0.3, "bathroom": 0.25, "living_room": 0.2, "kitchen": 0.15, "hallway": 0.1},
    "bathroom": {"hallway": 0.4, "bedroom": 0.4, "bathroom": 0.2},
    "unknown": {"hallway": 0.2, "bedroom": 0.2, "living_room": 0.2, "kitchen": 0.2, "bathroom": 0.2},
}

# Object co-occurrence: P(target | observed_object_nearby)
# When we see object X nearby, how much does it boost the probability of target Y?
OBJECT_COOCCURRENCE = {
    "toilet": {"sink": 0.85, "bathtub": 0.9, "shower": 0.9, "towel": 0.7},
    "bed": {"chest_of_drawers": 0.85, "towel": 0.3},
    "chair": {"table": 0.7, "sofa": 0.5, "tv_monitor": 0.4},
    "tv_monitor": {"sofa": 0.8, "chair": 0.5, "fireplace": 0.4},
    "sofa": {"tv_monitor": 0.7, "chair": 0.5, "fireplace": 0.4, "table": 0.3},
    "plant": {"chair": 0.3, "sofa": 0.3, "table": 0.3, "stairs": 0.4},
}

# Object co-occurrence priors: P(target | room_type)
# Based on common indoor layouts
TARGET_ROOM_PRIOR = {
    "chair": {"living_room": 0.5, "kitchen": 0.3, "bedroom": 0.1, "hallway": 0.05, "bathroom": 0.05},
    "bed": {"bedroom": 0.9, "living_room": 0.05, "hallway": 0.02, "kitchen": 0.02, "bathroom": 0.01},
    "plant": {"living_room": 0.4, "hallway": 0.3, "kitchen": 0.15, "bedroom": 0.1, "bathroom": 0.05},
    "toilet": {"bathroom": 0.9, "hallway": 0.05, "bedroom": 0.02, "kitchen": 0.02, "living_room": 0.01},
    "tv_monitor": {"living_room": 0.7, "bedroom": 0.2, "kitchen": 0.05, "hallway": 0.03, "bathroom": 0.02},
    "sofa": {"living_room": 0.8, "bedroom": 0.1, "hallway": 0.05, "kitchen": 0.03, "bathroom": 0.02},
}


def enrich_frontiers(full_map_pred, frontier_points, frontier_areas, goal_name, radius=30):
    """For each frontier, find nearby objects and infer room type + target prior.

    Args:
        full_map_pred: semantic map tensor [C, H, W], channels 4+ are semantic categories
        frontier_points: list of [y, x] centroids
        frontier_areas: list of pixel counts
        goal_name: target object name (e.g. "toilet")
        radius: search radius in map pixels around frontier centroid

    Returns:
        List of dicts with enriched info per frontier
    """
    semantic_map = full_map_pred[4:]  # [15, H, W]
    H, W = semantic_map.shape[1], semantic_map.shape[2]
    enriched = []

    for idx, (fy, fx) in enumerate(frontier_points):
        # Get nearby objects within radius
        y_min, y_max = max(0, fy - radius), min(H, fy + radius)
        x_min, x_max = max(0, fx - radius), min(W, fx + radius)
        patch = semantic_map[:, y_min:y_max, x_min:x_max]

        nearby_objects = []
        for cat_idx in range(len(hm3d_category)):
            if cat_idx < patch.shape[0] and patch[cat_idx].sum() > 2:
                nearby_objects.append(hm3d_category[cat_idx])

        # Infer room type from nearby objects
        room_scores = {}
        nearby_set = set(nearby_objects)
        for room, signature in ROOM_SIGNATURES.items():
            overlap = nearby_set & signature
            room_scores[room] = len(overlap) / max(len(signature), 1)

        best_room = max(room_scores, key=room_scores.get) if any(v > 0 for v in room_scores.values()) else "unknown"
        room_conf = room_scores.get(best_room, 0)

        # Target prior based on room type
        target_priors = TARGET_ROOM_PRIOR.get(goal_name, {})
        target_prior = target_priors.get(best_room, 0.1) if best_room != "unknown" else 0.1

        # Boost prior using object co-occurrence (even if room is "unknown")
        cooccurrence = OBJECT_COOCCURRENCE.get(goal_name, {})
        for obj in nearby_objects:
            if obj in cooccurrence:
                target_prior = max(target_prior, cooccurrence[obj])

        # FIX 1: For "unknown" rooms, use adjacency prior from nearby explored rooms
        # If we've explored a bedroom and this frontier is nearby, it might lead to bathroom
        if best_room == "unknown" and nearby_objects:
            # Check what rooms we've already identified in the explored area
            # Use a larger radius to find explored rooms
            big_radius = radius * 2
            by_min, by_max = max(0, fy - big_radius), min(H, fy + big_radius)
            bx_min, bx_max = max(0, fx - big_radius), min(W, fx + big_radius)
            big_patch = semantic_map[:, by_min:by_max, bx_min:bx_max]
            big_nearby = set()
            for ci in range(min(len(hm3d_category), big_patch.shape[0])):
                if big_patch[ci].sum() > 5:
                    big_nearby.add(hm3d_category[ci])

            # Infer what explored room is adjacent
            adj_room_scores = {}
            for room, sig in ROOM_SIGNATURES.items():
                overlap = big_nearby & sig
                if overlap:
                    adj_room_scores[room] = len(overlap)
            if adj_room_scores:
                explored_room = max(adj_room_scores, key=adj_room_scores.get)
                # Use adjacency to predict what's behind this unknown frontier
                adj_priors = ROOM_ADJACENCY.get(explored_room, {})
                for adj_room, adj_prob in adj_priors.items():
                    room_target_prior = target_priors.get(adj_room, 0.05)
                    adj_boost = adj_prob * room_target_prior
                    target_prior = max(target_prior, adj_boost)
                    # If adjacent room is likely bathroom and we're looking for toilet
                    if adj_room == "bathroom" and goal_name == "toilet" and adj_prob > 0.2:
                        target_prior = max(target_prior, 0.5)

        # FIX 2: For frontier areas that are small and enclosed (small area) near bedrooms,
        # boost bathroom probability (bathrooms are small rooms near bedrooms)
        frontier_area = frontier_areas[idx] if idx < len(frontier_areas) else 0
        if goal_name == "toilet" and frontier_area < 20 and best_room == "unknown":
            # Small unexplored areas near explored regions are often bathrooms
            target_prior = max(target_prior, 0.35)

        enriched.append({
            "idx": idx,
            "centroid": (fy, fx),
            "area": frontier_areas[idx] if idx < len(frontier_areas) else 0,
            "nearby_objects": nearby_objects,
            "room_type": best_room if room_conf > 0 else "unknown",
            "room_confidence": round(room_conf, 2),
            "target_prior": round(target_prior, 2),
        })

    return enriched


def Objects_Extract(full_map_pred):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            # contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) > 30:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                Object_list[hm3d_category[i]] = Single_object_list

    return Object_list

def form_prompt_for_chatgpt(goal_name, pose_pred, object_list, Wall_list, Frontier_list, last_decision, Frontier_points, enriched_frontiers=None):

    Robot_Position = "\n ".join([f"robot_{i}: {pose_pred[i][0], pose_pred[i][1]}" 
                        for i in range(len(pose_pred))])

    for key, value in object_list.items():
        value = ", ".join([f"{value[i]}" 
                        for i in range(len(value))])
    Objects_Position = "\n ".join([f"{key}: " + ", ".join([f"<" + ", ".join([f"{value[i][j][0][0], value[i][j][0][1]}" 
                            for j in range(len(value[i]))]) + f">"
                            for i in range(len(value))])
                            for key, value in object_list.items()]) + "\n"

    if Wall_list is not None:
        Walls_Position = "\n ".join([f"wall_{i}: <" +  ", ".join([f"{Wall_list[i][0][j]}"  
                            for j in range(len(Wall_list[i][0]))]) + f">"
                            for i in range(len(Wall_list))]) + "\n" 
    else: 
        Walls_Position = None

    if enriched_frontiers:
        # Sort by target_prior descending (highest likelihood first, not largest area first)
        sorted_frontiers = sorted(enriched_frontiers, key=lambda e: -e["target_prior"])
        frontier_strs = []
        for ef in sorted_frontiers:
            i = ef["idx"]
            parts = [f"centroid: {ef['centroid'][0], ef['centroid'][1]}", f"number: {ef['area']}"]
            if ef["room_type"] != "unknown":
                parts.append(f"room_type: {ef['room_type']}")
            if ef["nearby_objects"]:
                parts.append(f"nearby: {', '.join(ef['nearby_objects'][:5])}")
            parts.append(f"target_likelihood: {ef['target_prior']}")
            frontier_strs.append(f"frontier_{i}: <{', '.join(parts)}>")
        Frontiers = "\n ".join(frontier_strs)
    else:
        Frontiers = "\n ".join([f"frontier_{i}: <centroid: {Frontier_points[i][0], Frontier_points[i][1]}, number: {Frontier_list[i]}>"
                            for i in range(len(Frontier_points))])

    if len(last_decision) > 0:
        Last_Decision = "\n ".join([f"robot_{i}: {last_decision[i]}" 
                        for i in range(len(last_decision))])
    else:
        Last_Decision = "No frontiers"

    prompt_template = """
    [input:]
    Task: Locate the {GOAL_NAME}

    Position: 
    {ROBOT_POSITION}

    Scene Objects:
    {OBJECTS_POSITION}

    Walls:
    {WALLS_POSITION}

    Previous Movements:
    {LAST_DECISION}

    Unexplored Frontier: 
    {FRONTIERS}
                
    [output:] """

    User_prompt = prompt_template.format(
                    GOAL_NAME = str(goal_name),
                    ROBOT_POSITION = Robot_Position,
                    OBJECTS_POSITION = Objects_Position,
                    WALLS_POSITION = Walls_Position,
                    FRONTIERS = Frontiers,
                    LAST_DECISION = Last_Decision
                )

    Frontiers_dict = {}
    for i in range(len(Frontier_points)):
        Frontiers_dict['frontier_' + str(i)] = f"<centroid: {Frontier_points[i][0], Frontier_points[i][1]}, number: {Frontier_list[i]}>"

    return User_prompt, Frontiers_dict

# def parse_answer(response_message):
#     lines = response_message.split('\n')

#     parsed_dict_num = {}

#     for line in lines:
#         key, value = line.split(': ')
#         parsed_dict_num[key.strip()] = int(value.split('_')[1])

#     return parsed_dict_num


def parse_answer(response_message):
    import re
    lines = response_message.split('\n')
    parsed_dict_num = {}

    for line in lines:
        line = line.strip()
        # Skip empty lines and the [output:] line
        if not line or line.startswith('[output:]'):
            continue

        # Look for lines that contain robot_ and frontier_ assignments
        # Handle noisy formats: "robot_0:! frontier_1", "robot_1!! frontier_3", etc.
        if 'robot_' in line and 'frontier_' in line:
            try:
                # Extract robot id and frontier id with regex
                match = re.search(r'(robot_\d+)\s*[:\!]+\s*frontier_(\d+)', line)
                if match:
                    key = match.group(1)
                    frontier_num = int(match.group(2))
                    parsed_dict_num[key] = frontier_num
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue

    return parsed_dict_num


@habitat.registry.register_task_action
class TurnLeftAction_S(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.TURN_LEFT_S)


@habitat.registry.register_task_action
class TurnRightAction_S(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.TURN_RIGHT_S)


@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config

def main():
    args = get_args()
    if (args.kg_memory_mode == "scene"
            and args.kg_coordinate_frame != "world"):
        raise ValueError(
            "--kg_memory_mode scene requires --kg_coordinate_frame world"
        )

    brain_config = load_brain_api_config(PROJECT_ROOT)
    if brain_config.backend == "vllm":
        if brain_config.base_url.startswith(
            ("http://127.0.0.1", "http://localhost")
        ):
            for proxy_bypass_var in ("NO_PROXY", "no_proxy"):
                bypass_hosts = os.environ.get(proxy_bypass_var, "").split(",")
                bypass_hosts = [host for host in bypass_hosts if host]
                for host in ("127.0.0.1", "localhost"):
                    if host not in bypass_hosts:
                        bypass_hosts.append(host)
                os.environ[proxy_bypass_var] = ",".join(bypass_hosts)
        print(
            "Using vLLM brain service: "
            f"{brain_config.base_url} ({brain_config.model})"
        )
        print("Skipping in-process Qwen loading; vLLM owns the model instance.")
    elif brain_config.backend == "deepseek":
        thinking = brain_config.extra_body["thinking"]["type"]
        print(
            "Using DeepSeek API brain: "
            f"{brain_config.base_url} ({brain_config.model}, "
            f"thinking={thinking})"
        )
        print("Skipping in-process Qwen loading; DeepSeek is API-hosted.")
    elif brain_config.backend == "qqqapi":
        reasoning_effort = brain_config.extra_body["reasoning_effort"]
        print(
            "Using qqqapi API brain: "
            f"{brain_config.base_url} ({brain_config.model}, "
            f"reasoning_effort={reasoning_effort})"
        )
        print("Skipping in-process Qwen loading; qqqapi is API-hosted.")
    else:
        # Preserve the original local-model initialization outside vLLM mode.
        model_path = args.llm_path or LOCAL_MODEL_PATHS[args.gpt_type]
        model_type = "vl" if "VL" in model_path else "text"
        llm_gpu_id = args.llm_gpu_id if args.llm_gpu_id >= 0 else args.sem_gpu_id
        vlm_device = f"cuda:{llm_gpu_id}"
        load_model(model_path, device=vlm_device, model_type=model_type)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()

    # The audited two-agent configuration remains the untouched default.
    # Only the isolated robot-count ablation launcher passes 1 or 3 here.
    robot_count_runtime = None
    if args.num_agents != 2:
        import robot_count_runtime as robot_count_runtime_module
        robot_count_runtime = robot_count_runtime_module
        robot_count_runtime.configure_habitat_agents(
            config_env, args.num_agents
        )

    # Apply --split argument to dataset config
    config_env.DATASET.SPLIT = args.split
    config_env.DATASET.DATA_PATH = config_env.DATASET.DATA_PATH.replace("{split}", args.split)
    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id
    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
    episode_shuffle = os.environ.get("EPISODE_SHUFFLE")
    if episode_shuffle is not None:
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = episode_shuffle.lower() in {
            "1", "true", "yes", "on"
        }

    config_env.TASK.POSSIBLE_ACTIONS = config_env.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    config_env.freeze()


    env = Multi_Agent_Env(
        config_env=config_env,
        use_gtsem=bool(args.use_gtsem),
    )

    total_episodes = env.number_of_episodes
    if args.start_episode_index < 0 or args.start_episode_index >= total_episodes:
        raise ValueError(
            f"start_episode_index must be in [0, {total_episodes - 1}], "
            f"got {args.start_episode_index}"
        )

    for _ in range(args.start_episode_index):
        env.reset()

    num_episodes = total_episodes - args.start_episode_index

    if args.max_episodes > 0:
        num_episodes = min(num_episodes, args.max_episodes)

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS
    agent = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, i))


    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'output.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    logging.info(args)
    mapping_jsonl_log = None
    kg_jsonl_log = None
    scene_kg_jsonl_log = None
    if args.jsonl_log:
        jsonl_dir = os.path.dirname(os.path.abspath(args.jsonl_log))
        os.makedirs(jsonl_dir, exist_ok=True)
        open(args.jsonl_log, "w").close()
        mapping_jsonl_log = os.path.splitext(args.jsonl_log)[0] + ".mapping.jsonl"
        kg_jsonl_log = os.path.splitext(args.jsonl_log)[0] + ".kg.jsonl"
        scene_kg_jsonl_log = (
            os.path.splitext(args.jsonl_log)[0] + ".scene-kg.jsonl"
        )
        open(mapping_jsonl_log, "w").close()
        open(kg_jsonl_log, "w").close()
        open(scene_kg_jsonl_log, "w").close()
        print(f"Writing per-episode JSONL to {args.jsonl_log}")
        print(f"Writing room/frontier mapping audit to {mapping_jsonl_log}")
        print(f"Writing complete KG snapshots to {kg_jsonl_log}")
    if args.stop_diag_jsonl:
        stop_diag_dir = os.path.dirname(
            os.path.abspath(args.stop_diag_jsonl)
        )
        os.makedirs(stop_diag_dir, exist_ok=True)
        open(args.stop_diag_jsonl, "w").close()
        print(f"Writing per-step STOP diagnostics to {args.stop_diag_jsonl}")
    target_diag_goals = {
        goal.strip()
        for goal in args.target_diag_goals.split(",")
        if goal.strip()
    }
    if args.target_diag_dir:
        os.makedirs(os.path.abspath(args.target_diag_dir), exist_ok=True)
        print(
            "Writing target diagnostic artifacts to "
            f"{os.path.abspath(args.target_diag_dir)} "
            f"for goals={sorted(target_diag_goals)}"
        )
    # ------------------------------------------------------------------


    device = torch.device("cuda:0" if args.cuda else "cpu")

    agg_metrics: Dict = defaultdict(float)

    count_episodes = 0
    count_step = 0
    goal_points = []
    log_start = time.time()
    last_decision = []
    total_usage = []
    decision_history = []
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    brain_adapter = OpenAICompatibleBrainAdapter(
        config=brain_config,
        usage_sink=total_usage,
        seed=args.seed if args.reset_seed_each_episode else None,
    )
    kg = KnowledgeGraph()
    scene_kgs = {}
    # This is deliberately separate from node provenance.  A valid task can
    # finish without leaving a static node, but it still counts as a prior
    # same-scene task for the next episode's memory audit.
    scene_completed_episode_counts = defaultdict(int)
    brain_class = HelicaseBrain
    if num_agents != 2:
        from brain_robot_ablation import RobotCountAblationBrain
        brain_class = RobotCountAblationBrain
    mindnav_brain = brain_class(
        brain_adapter,
        num_agents=num_agents,
        kg_serialization=args.kg_serialization,
        decision_history_enabled=args.decision_history == "on",
    )
    print(f"Using MindNav core: {CODE_SRC}/brain.py + {CODE_SRC}/kg_construction.py")

    while count_episodes < num_episodes:
        episode_started_at = datetime.now(timezone.utc)
        usage_before_episode = brain_adapter.usage_snapshot()
        episode_diagnostics = defaultdict(int, {
            "invalid_outputs": 0,
            "repair_calls": 0,
            "outer_retries": 0,
            "fallback_decisions": 0,
            "planning_rounds": 0,
        })
        episode_errors = []
        episode_seed = None
        if args.reset_seed_each_episode:
            episode_seed = reset_episode_rng(args.seed, env=env)
            print(
                f"[EPISODE_SEED] index="
                f"{args.start_episode_index + count_episodes} "
                f"seed={episode_seed} mode=fixed_global"
            )
        observations = env.reset()
        current_episode = env.current_episode
        scene_key = os.path.basename(current_episode.scene_id)
        frame = None
        if args.kg_coordinate_frame == "world":
            frame = EpisodeFrame(
                scene_id=scene_key,
                start_position=tuple(
                    float(value) for value in current_episode.start_position
                ),
                start_rotation=tuple(
                    float(value) for value in current_episode.start_rotation
                ),
                map_size_px=MAP_SIZE,
                map_resolution_m=float(args.map_resolution) / 100.0,
            )
        if args.kg_memory_mode == "scene":
            kg = scene_kgs.setdefault(scene_key, KnowledgeGraph())
            kg.reset_episode_state()
            scene_memory_previous_episodes = scene_completed_episode_counts[
                scene_key
            ]
            scene_memory_nodes_before = len(kg.nodes)
            scene_memory_edges_before = len(kg.edges)
        else:
            kg.reset()
            scene_memory_previous_episodes = 0
            scene_memory_nodes_before = 0
            scene_memory_edges_before = 0
        episode_target_name = str(
            getattr(current_episode, "object_category", "")
        )
        historical_target_node_ids = {
            node.id for node in kg.get_nodes_by_type("object")
            if node.name == episode_target_name
        }
        historical_target_room_ids = {
            edge.source for edge in kg.edges
            if edge.relation == "contains"
            and edge.target in historical_target_node_ids
        }
        historical_target_seen = bool(historical_target_node_ids)
        kg_updater = KGUpdater(
            kg,
            frame=frame,
            persistent_scene=args.kg_memory_mode == "scene",
            episode_id=str(current_episode.episode_id),
        )
        alignment_errors_m = []
        initial_agent_poses = None
        if num_agents != 2:
            initial_agent_poses = robot_count_runtime.capture_initial_agent_poses(
                env, num_agents
            )
        print(
            f"[EPISODE_START] index={args.start_episode_index + count_episodes} "
            f"episode_id={current_episode.episode_id} "
            f"scene={os.path.basename(current_episode.scene_id)} "
            f"target={getattr(current_episode, 'object_category', 'unknown')}"
        )
        decision_history.clear()
        last_decision.clear()
        goal_points.clear()
        for i in range(num_agents):
            agent[i].reset()
        target_first_found_saved = [False] * num_agents

        while not env.episode_over:
            if num_agents == 2:
                action = [0, 0]
            else:
                action = robot_count_runtime.initialize_actions(num_agents)
            full_map = []
            visited_vis = []
            pose_pred = []
            start = time.time()
            for i in range(num_agents):
                agent[i].mapping(observations[i])
                full_map.append(agent[i].local_map)
                visited_vis.append(agent[i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[i].visited_vis.shape[0],
                    (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)
                if frame is not None:
                    try:
                        map_rc = CurrentFrontierView.robot_pose_to_map_rc(
                            pos, MAP_SIZE
                        )
                        predicted_xz = frame.map_rc_to_world_xz(*map_rc)
                        agent_state = env.sim.get_agent_state(i)
                        actual_xz = (
                            float(agent_state.position[0]),
                            float(agent_state.position[2]),
                        )
                        alignment_errors_m.append(float(np.hypot(
                            predicted_xz[0] - actual_xz[0],
                            predicted_xz[1] - actual_xz[1],
                        )))
                    except Exception as alignment_error:
                        episode_errors.append(
                            f"alignment_audit: {alignment_error}"
                        )
                
            if num_agents == 2:
                full_map2 = torch.cat((full_map[0].unsqueeze(0), full_map[1].unsqueeze(0)), 0)
                full_map_pred, _ = torch.max(full_map2, 0)
            else:
                full_map_pred = robot_count_runtime.fuse_agent_maps(full_map)

            # mapping_end = time.time()
            # mapping_time = mapping_end - start
            # print('mapping_time: %.3f秒'%mapping_time)

            if agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0:
                goal_points.clear()

                Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred)

                if len(target_point_map) > 0:
                    object_list = Objects_Extract(full_map_pred)

                    # Enrich frontiers with room-type + nearby objects + target prior
                    enriched = enrich_frontiers(full_map_pred, target_point_map, Frontier_list, agent[0].goal_name)

                    # ── 025 Probe: log frontier context for analysis ──
                    probe_record = {
                        "step": agent[0].l_step,
                        "goal": agent[0].goal_name,
                        "num_frontiers": len(target_point_map),
                        "frontiers": [{"idx": e["idx"], "room": e["room_type"], "nearby": e["nearby_objects"],
                                       "prior": e["target_prior"], "area": e["area"]} for e in enriched],
                    }

                    Frontiers_dict = {
                        "frontier_" + str(i): (
                            f"<centroid: {target_point_map[i][0], target_point_map[i][1]}, "
                            f"number: {Frontier_list[i]}>"
                        )
                        for i in range(len(target_point_map))
                    }

                    # Build the executable mapping for this decision only.
                    # The persistent KG update below deliberately does not
                    # retain frontier indices.
                    current_frontiers = (
                        kg_updater.build_current_frontier_view(enriched)
                    )
                    for frontier_idx in current_frontiers.frontiers:
                        if not 0 <= frontier_idx < len(target_point_map):
                            raise ValueError(
                                f"Current frontier_{frontier_idx} has no "
                                "matching target point"
                            )

                    kg_updater.update(
                        enriched,
                        object_list,
                        pose_pred,
                        Wall_list,
                    )
                    if kg_jsonl_log:
                        kg_snapshot = build_kg_snapshot_record(
                            args.start_episode_index + count_episodes,
                            current_episode.episode_id,
                            current_episode.scene_id,
                            agent[0].goal_name,
                            agent[0].l_step,
                            kg,
                            "decision",
                            frame=frame,
                            memory_mode=args.kg_memory_mode,
                        )
                        with open(kg_jsonl_log, "a") as kg_file:
                            json.dump(
                                kg_snapshot,
                                kg_file,
                                ensure_ascii=False,
                            )
                            kg_file.write("\n")

                    retries = 3
                    vlm_success = False
                    goal_frontiers = {}
                    tools_called = []
                    decision_method = ""
                    last_llm_response = ""
                    last_brain_error = None
                    while retries > 0:
                        try:
                            goal_frontiers, tools_called, decision_method = mindnav_brain.decide(
                                kg,
                                agent[0].goal_name,
                                enriched,
                                pose_pred,
                                agent[0].l_step,
                                args.max_episode_length,
                                decision_history,
                                current_frontiers=current_frontiers,
                                current_episode_id=str(current_episode.episode_id),
                            )
                            last_llm_response = brain_adapter.last_response
                            print(
                                f"MindNav decision: {decision_method}; tools={tools_called}; "
                                f"kg_nodes={len(kg.nodes)}, kg_edges={len(kg.edges)}"
                            )

                            # Verify we got assignments for all robots
                            for i in range(num_agents):
                                robot_key = "robot_" + str(i)
                                if robot_key not in goal_frontiers:
                                    raise ValueError(f"Missing frontier assignment for {robot_key}")
                                frontier_idx = int(goal_frontiers[robot_key])
                                if frontier_idx not in current_frontiers.frontiers:
                                    raise ValueError(
                                        f"Non-current frontier_{frontier_idx} "
                                        f"assigned to {robot_key}"
                                    )
                                goal_frontiers[robot_key] = frontier_idx

                            if len(current_frontiers.frontiers) >= num_agents:
                                assigned_indices = list(goal_frontiers.values())
                                if len(assigned_indices) != len(set(assigned_indices)):
                                    raise ValueError(
                                        "Duplicate frontier assignments despite "
                                        "enough current candidates"
                                    )

                            vlm_success = True
                            break
                        except Exception as e:
                            last_brain_error = e
                            episode_diagnostics["outer_retries"] += 1
                            episode_errors.append(
                                f"{type(e).__name__}: {str(e)[:500]}"
                            )
                            print(f"MindNav core brain error: {e}")
                            print('Retrying...')
                            retries -= 1
                            time.sleep(1)

                    if not vlm_success:
                        reason = (
                            type(last_brain_error).__name__
                            if last_brain_error is not None else "brain_error"
                        )
                        # Transport failures and unexpected exceptions can
                        # exhaust the outer retry loop before ``decide``
                        # returns a complete two-robot assignment.  They must
                        # degrade to the same current-frontier-only fallback
                        # as validated model-output failures, not fall through
                        # with a partial/stale assignment and abort the run.
                        goal_frontiers, tools_called, decision_method = (
                            mindnav_brain.deterministic_fallback(
                                current_frontiers,
                                pose_pred,
                                reason=reason,
                            )
                        )

                    decision_audit = mindnav_brain.last_decision_audit
                    episode_diagnostics["planning_rounds"] += 1
                    episode_diagnostics["invalid_outputs"] += len(
                        decision_audit.get("validation_errors", [])
                    )
                    episode_diagnostics["repair_calls"] += sum(
                        status == "repaired"
                        for status in decision_audit.get(
                            "tool_status", {}
                        ).values()
                    )
                    fallback_reason = decision_audit.get("fallback_reason")
                    if fallback_reason:
                        # The brain can return a valid deterministic fallback
                        # directly, in which case the outer retry loop marks
                        # the call successful and the legacy ``reason`` local
                        # above is never assigned.  Use the audited fallback
                        # reason as the single source of truth.
                        reason = fallback_reason
                        episode_diagnostics["fallback_decisions"] += 1
                        print(
                            "MindNav failed all retries; using deterministic "
                            f"current-frontier fallback ({reason})"
                        )
                        goal_frontiers, tools_called, decision_method = (
                            mindnav_brain.deterministic_fallback(
                                current_frontiers,
                                pose_pred,
                                reason=reason,
                            )
                        )

                    # One strict execution path for both LLM and fallback
                    # decisions. No index clamping or post-hoc redirection.
                    for i in range(num_agents):
                        robot_key = "robot_" + str(i)
                        if robot_key not in goal_frontiers:
                            raise ValueError(
                                f"Missing final assignment for {robot_key}"
                            )
                        frontier_idx = int(goal_frontiers[robot_key])
                        if frontier_idx not in current_frontiers.frontiers:
                            raise ValueError(
                                f"Final assignment frontier_{frontier_idx} "
                                "is not current"
                            )
                        goal_frontiers[robot_key] = frontier_idx

                    if len(current_frontiers.frontiers) >= num_agents:
                        final_indices = list(goal_frontiers.values())
                        if len(final_indices) != len(set(final_indices)):
                            raise ValueError(
                                "Final current-frontier assignments are not unique"
                            )


                    last_decision.clear()
                    for i in range(num_agents):
                        fi = goal_frontiers["robot_" + str(i)]
                        goal_points.append(target_point_map[fi])
                        last_decision.append(
                            Frontiers_dict.get("frontier_" + str(fi), "")
                        )

                    if args.decision_history == "on":
                        if num_agents == 2:
                            decision_history.append({
                                "step": int(agent[0].l_step),
                                "r0": int(goal_frontiers.get("robot_0", 0)),
                                "r1": int(goal_frontiers.get("robot_1", 0)),
                                "objects_found": ",".join(
                                    sorted(object_list.keys())
                                ),
                            })
                        else:
                            decision_history.append(
                                robot_count_runtime.build_history_record(
                                    agent[0].l_step,
                                    goal_frontiers,
                                    object_list.keys(),
                                    num_agents,
                                )
                            )

                    if num_agents == 2:
                        chosen_0 = goal_frontiers.get("robot_0", 0)
                        chosen_1 = goal_frontiers.get("robot_1", 0)
                        probe_record["chosen_0"] = chosen_0
                        probe_record["chosen_1"] = chosen_1
                        probe_record["same_frontier"] = (chosen_0 == chosen_1)
                        print(
                            f"  [PROBE] step={agent[0].l_step}, "
                            f"goal={agent[0].goal_name}, "
                            f"chose=({chosen_0},{chosen_1}), "
                            f"same={'Y' if chosen_0 == chosen_1 else 'N'}, "
                            f"rooms={[e['room_type'] for e in enriched]}, "
                            f"priors={[e['target_prior'] for e in enriched]}"
                        )
                    else:
                        probe_fields = robot_count_runtime.build_probe_fields(
                            goal_frontiers, num_agents
                        )
                        probe_record.update(probe_fields)
                        print(
                            f"  [PROBE] step={agent[0].l_step}, "
                            f"goal={agent[0].goal_name}, "
                            f"chose={probe_fields['chosen_frontiers']}, "
                            f"duplicates="
                            f"{probe_fields['duplicate_frontier_count']}, "
                            f"rooms={[e['room_type'] for e in enriched]}, "
                            f"priors={[e['target_prior'] for e in enriched]}"
                        )

                    if mapping_jsonl_log:
                        try:
                            mapping_audit = build_room_frontier_audit(
                                args.start_episode_index + count_episodes,
                                current_episode.episode_id,
                                agent[0].l_step,
                                agent[0].goal_name,
                                enriched,
                                current_frontiers,
                                kg,
                                last_llm_response,
                                goal_frontiers,
                                mindnav_brain.last_decision_audit,
                                decision_method,
                                tools_called,
                            )
                            with open(mapping_jsonl_log, "a") as mapping_file:
                                json.dump(
                                    mapping_audit,
                                    mapping_file,
                                    ensure_ascii=False,
                                )
                                mapping_file.write("\n")
                            audit_counts = mapping_audit["counts"]
                            print(
                                "  [MAPPING] "
                                f"active={audit_counts['active_frontiers']}, "
                                f"rooms={audit_counts['unique_current_rooms']}, "
                                f"collisions={audit_counts['collision_rooms']}, "
                                f"stale_candidates="
                                f"{audit_counts['stale_candidate_rooms']}, "
                                f"executed_noncurrent="
                                f"{audit_counts['executed_noncurrent_frontiers']}, "
                                f"mismatches="
                                f"{audit_counts['room_frontier_mismatches']}, "
                                f"persistent_transient="
                                f"{audit_counts['persistent_transient_room_properties']}"
                            )
                        except Exception as audit_error:
                            print(
                                f"  [MAPPING] audit logging failed: "
                                f"{audit_error}"
                            )
                else:
                    for i in range(num_agents):
                        actions = np.random.rand(1, 2).squeeze()*(target_edge_map.shape[0] - 1)

                        goal_points.append([int(actions[0]), int(actions[1])])

            # start_act = time.time()
            for i in range(num_agents):
                action[i] = agent[i].act(goal_points[i])
            # act_end = time.time()
            # act_time = act_end - start_act
            # print('act_time: %.3f秒'%act_time)


            observations = env.step(action)

            if args.stop_diag_jsonl or args.target_diag_dir:
                step_metrics = env.get_metrics()
                per_agent_dtg = get_per_agent_goal_distances(
                    env,
                    current_episode,
                    num_agents,
                )
                target_artifacts = []
                if (
                    args.target_diag_dir
                    and agent[0].goal_name in target_diag_goals
                ):
                    for i in range(num_agents):
                        events = []
                        if (
                            agent[i].last_found_goal
                            and not target_first_found_saved[i]
                        ):
                            events.append("first_found")
                            target_first_found_saved[i] = True
                        if int(action[i]) == 0:
                            events.append("stop")
                        for event in events:
                            artifact = save_target_diagnostic_artifact(
                                agent=agent[i],
                                base_dir=args.target_diag_dir,
                                episode_index=(
                                    args.start_episode_index + count_episodes
                                ),
                                goal=agent[0].goal_name,
                                step=int(agent[0].l_step),
                                event=event,
                                action=action[i],
                                global_distance_to_goal=float(
                                    step_metrics.get(
                                        "distance_to_goal", -1.0
                                    )
                                ),
                                agent_distance_to_goal=per_agent_dtg[i],
                            )
                            if artifact:
                                target_artifacts.append(artifact)
            if args.stop_diag_jsonl:
                stop_record = {
                    "schema_version": 1,
                    "episode_index": int(
                        args.start_episode_index + count_episodes
                    ),
                    "episode": int(
                        args.start_episode_index + count_episodes + 1
                    ),
                    "episode_id": str(current_episode.episode_id),
                    "goal": agent[0].goal_name,
                    "step": int(agent[0].l_step),
                    "global_distance_to_goal": float(
                        step_metrics.get("distance_to_goal", -1.0)
                    ),
                    "per_agent_distance_to_goal": per_agent_dtg,
                    "success": float(step_metrics.get("success", 0.0)),
                    "episode_over": bool(env.episode_over),
                    "any_stop_action": any(int(a) == 0 for a in action),
                    "robots": [
                        {
                            "robot_id": f"robot_{i}",
                            "action": int(action[i]),
                            "found_goal": int(agent[i].last_found_goal),
                            "planner_stop": bool(agent[i].last_planner_stop),
                            "target_map_mass": float(
                                agent[i].last_target_map_mass
                            ),
                            "goal_map_mass": float(
                                agent[i].last_goal_map_mass
                            ),
                            "replan_count": int(agent[i].replan_count),
                            "semantic_sources": (
                                agent[i].last_semantic_source_stats
                            ),
                            "target_map_stats": (
                                agent[i].last_target_map_stats
                            ),
                            "pose": [
                                float(value)
                                for value in agent[i].planner_pose_inputs[:3]
                            ],
                        }
                        for i in range(num_agents)
                    ],
                    "target_artifacts": target_artifacts,
                }
                with open(args.stop_diag_jsonl, "a") as stop_diag_file:
                    json.dump(stop_record, stop_diag_file, ensure_ascii=False)
                    stop_diag_file.write("\n")
            # step_end = time.time()
            # step_time = step_end - act_end
            # print('step_time: %.3f秒'%step_time)


            if args.visualize or args.print_images: 
                Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                            agent[0].goal_id, visited_vis, target_edge_map, goal_points)

        
        count_episodes += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### Logging
        # ------------------------------------------------------------------
        log_end = time.time()
        time_elapsed = time.gmtime(log_end - log_start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_step),
            "FPS {},".format(int(count_step / (log_end - log_start)))
        ]) + '\n'

        metrics = env.get_metrics()
        if frame is not None:
            median_alignment_error = (
                float(np.median(alignment_errors_m))
                if alignment_errors_m else float("inf")
            )
            max_alignment_error = (
                float(np.max(alignment_errors_m))
                if alignment_errors_m else float("inf")
            )
            if (median_alignment_error > args.kg_alignment_median_threshold_m
                    or max_alignment_error > args.kg_alignment_max_threshold_m):
                raise RuntimeError(
                    "KG world-coordinate alignment failed: "
                    f"median={median_alignment_error:.4f} m "
                    f"(limit={args.kg_alignment_median_threshold_m:.4f}), "
                    f"max={max_alignment_error:.4f} m "
                    f"(limit={args.kg_alignment_max_threshold_m:.4f})"
                )
        if kg_jsonl_log:
            final_kg_snapshot = build_kg_snapshot_record(
                args.start_episode_index + count_episodes - 1,
                current_episode.episode_id,
                current_episode.scene_id,
                agent[0].goal_name,
                agent[0].l_step,
                kg,
                "final",
                metrics=metrics,
                frame=frame,
                memory_mode=args.kg_memory_mode,
            )
            with open(kg_jsonl_log, "a") as kg_file:
                json.dump(
                    final_kg_snapshot,
                    kg_file,
                    ensure_ascii=False,
                )
                kg_file.write("\n")
        if scene_kg_jsonl_log:
            # This audit stream represents reusable scene memory, not the
            # live episode graph.  Strip robots and all transient execution
            # relations from a copy so it cannot be mistaken for history.
            scene_kg_for_snapshot = copy.deepcopy(kg)
            scene_kg_for_snapshot.reset_episode_state()
            scene_kg_snapshot = build_kg_snapshot_record(
                args.start_episode_index + count_episodes - 1,
                current_episode.episode_id,
                current_episode.scene_id,
                agent[0].goal_name,
                agent[0].l_step,
                scene_kg_for_snapshot,
                "final",
                metrics=metrics,
                frame=frame,
                memory_mode=args.kg_memory_mode,
            )
            with open(scene_kg_jsonl_log, "a") as scene_kg_file:
                json.dump(scene_kg_snapshot, scene_kg_file, ensure_ascii=False)
                scene_kg_file.write("\n")
        if args.jsonl_log:
            usage_after_episode = brain_adapter.usage_snapshot()
            episode_usage = {
                key: int(usage_after_episode[key] - usage_before_episode[key])
                for key in usage_after_episode
            }
            episode_ended_at = datetime.now(timezone.utc)
            episode_record = {
                "schema_version": 4,
                "run_id": args.method_name,
                "method": args.method_name,
                "kg_serialization": args.kg_serialization,
                "kg_coordinate_frame": args.kg_coordinate_frame,
                "kg_memory_mode": args.kg_memory_mode,
                "decision_history": args.decision_history,
                "semantic_source": (
                    "habitat_gt" if args.use_gtsem else "predicted"
                ),
                "semantic_pixel_counts_by_agent": (
                    [a.semantic_pixel_counts.tolist() for a in agent]
                    if args.use_gtsem else None
                ),
                "episode": args.start_episode_index + count_episodes,
                "episode_index": (
                    args.start_episode_index + count_episodes - 1
                ),
                "episode_id": str(current_episode.episode_id),
                "scene": os.path.basename(current_episode.scene_id),
                "start_position": [
                    float(value) for value in current_episode.start_position
                ],
                "start_rotation": [
                    float(value) for value in current_episode.start_rotation
                ],
                "floor_id": frame.floor_id if frame is not None else None,
                "scene_memory_previous_episodes": (
                    scene_memory_previous_episodes
                ),
                "scene_memory_nodes_before_episode": scene_memory_nodes_before,
                "scene_memory_edges_before_episode": scene_memory_edges_before,
                "scene_memory_nodes_after_episode": len(kg.nodes),
                "scene_memory_edges_after_episode": len(kg.edges),
                "alignment_robot_median_error_m": (
                    float(np.median(alignment_errors_m))
                    if alignment_errors_m else None
                ),
                "alignment_robot_max_error_m": (
                    float(np.max(alignment_errors_m))
                    if alignment_errors_m else None
                ),
                "historical_target_seen": historical_target_seen,
                "historical_target_room_count": int(
                    len(historical_target_room_ids)
                ),
                "success": float(metrics.get("success", 0.0)),
                "spl": float(metrics.get("spl", 0.0)),
                "habitat_success": float(metrics.get("success", 0.0)),
                "habitat_spl": float(metrics.get("spl", 0.0)),
                "goal": agent[0].goal_name,
                "steps": int(agent[0].l_step),
                "distance_to_goal": float(metrics.get("distance_to_goal", -1.0)),
                **episode_usage,
                **{key: int(value) for key, value in episode_diagnostics.items()},
                "backend": brain_config.backend,
                "base_url": brain_config.base_url,
                "model": brain_config.model,
                "habitat_sim_version": habitat_sim.__version__,
                "sim_gpu_id": int(args.sim_gpu_id),
                "sem_gpu_id": int(args.sem_gpu_id),
                "llm_gpu_id": int(args.llm_gpu_id),
                "success_threshold_m": float(
                    config_env.TASK.SUCCESS.SUCCESS_DISTANCE
                ),
                "episode_seed": episode_seed,
                "seed": int(args.seed),
                "seed_mode": (
                    "fixed_global"
                    if args.reset_seed_each_episode else "continuous"
                ),
                "git_commit": git_commit,
                "started_at": episode_started_at.isoformat(),
                "ended_at": episode_ended_at.isoformat(),
                "duration_seconds": (
                    episode_ended_at - episode_started_at
                ).total_seconds(),
                "status": "complete",
                "errors": episode_errors,
            }
            if num_agents != 2:
                episode_record.update({
                    "num_agents": int(num_agents),
                    "initial_agent_poses": initial_agent_poses,
                    "team_steps": int(agent[0].l_step),
                    "robot_actions": int(num_agents * agent[0].l_step),
                    "robot_count_ablation": True,
                })
            with open(args.jsonl_log, "a") as jsonl_file:
                json.dump(episode_record, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")
        if args.kg_memory_mode == "scene":
            scene_completed_episode_counts[scene_key] += 1
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        usage_totals = brain_adapter.usage_snapshot()
        log += " LLM usage: " + json.dumps(usage_totals, sort_keys=True)
        print(log)
        logging.info(log)
        # ------------------------------------------------------------------


    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    return avg_metrics

if __name__ == "__main__":
    main()
