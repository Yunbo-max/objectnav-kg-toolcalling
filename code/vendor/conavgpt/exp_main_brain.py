from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from agents.llm_agents import LLM_Agent
from agents.helicase_kg import KnowledgeGraph, KGUpdater
from agents.kg_logging import KGTraceLogger
from agents.mindnav_kg_brain import KGToolCallingBrain
from agents.mindnav_heuristic import MindNavHeuristicBrain
from envs.habitat.multi_agent_env import Multi_Agent_Env as PredMultiAgentEnv
from envs.habitat.multi_agent_sem_env import Multi_Agent_Env as GTMultiAgentEnv
from constants import color_palette
import utils.visualization as vu
from arguments import get_args

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

try:
    import yaml
except ImportError:
    yaml = None

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
LOCAL_MODEL_PATHS = [
    '/tf/notebooks/models/Qwen2.5-3B-Instruct',       # type 0: 3B
    '/tf/notebooks/models/Qwen2.5-7B-Instruct',       # type 1: 7B
]
gpt_name = ['Qwen2.5-3B', 'Qwen2.5-7B']


def default_mindnav_config_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/mindnav.yaml"))


def load_mindnav_config(path=None):
    cfg_path = path or default_mindnav_config_path()
    if not cfg_path or not os.path.exists(cfg_path):
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --mindnav_config")
    with open(cfg_path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return {}
    data["_config_path"] = cfg_path
    return data


def cfg_number(config, key, default, cast=float):
    try:
        return cast(config.get(key, default))
    except (TypeError, ValueError):
        return default


def cfg_bool(config, key, default=False):
    value = os.environ.get(key.upper(), config.get(key, default))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(default)


def is_mp3d_task(args):
    return getattr(args, "dataset", None) == "mp3d" or "objectnav_mp3d" in args.task_config


def semantic_categories(args):
    return args.semantic_categories


def goal_label(args, goal_name):
    if isinstance(goal_name, (int, np.integer)):
        goal_id = int(goal_name)
        if 0 <= goal_id < len(args.goal_id_to_name):
            return args.goal_id_to_name[goal_id]
        return str(goal_id)
    return str(goal_name)


def goal_channel(args, goal_name):
    goal_cat = goal_label(args, goal_name)
    goal_ch = args.category_to_channel.get(goal_cat)
    return goal_ch + 4 if goal_ch is not None else None


def sanitize_navigation_actions(args, actions):
    """Keep the ObjectNav loop on planar navigation actions."""
    disable_look = is_mp3d_task(args) or bool(int(os.environ.get("DISABLE_LOOK_ACTIONS", "0")))
    if not disable_look:
        return actions

    remapped = []
    for action in actions:
        if action == HabitatSimActions.LOOK_UP:
            remapped.append(HabitatSimActions.TURN_LEFT_S)
        elif action == HabitatSimActions.LOOK_DOWN:
            remapped.append(HabitatSimActions.TURN_RIGHT_S)
        else:
            remapped.append(action)
    return remapped


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

    semantic_scores = full_map_pred[4:, :, :]
    sem_map = semantic_scores.argmax(0).cpu().numpy()
    no_cat_mask = (torch.max(semantic_scores, dim=0)[0] <= 0).cpu().numpy()

    sem_map += 5

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
    cn = goal_channel(args, goal_name)
    if cn is not None and full_map_pred[cn, :, :].sum() != 0.:
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

    vis_image = vu.init_multi_vis_image(goal_label(args, goal_name), color)

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

# Room type inference from nearby objects. Values are evidence weights, not
# normalized signature membership; a single strong object can identify a room.
ROOM_SIGNATURES = {
    "bathroom": {"toilet": 1.0, "shower": 1.0, "bathtub": 1.0, "sink": 0.7, "towel": 0.7, "mirror": 0.8, "counter": 0.4, "cabinet": 0.3},
    "bedroom": {"bed": 1.0, "chest_of_drawers": 0.9, "drawer": 0.9, "clothes": 0.8, "curtain": 0.45, "blinds": 0.45, "cabinet": 0.4, "window": 0.35, "mirror": 0.35, "picture": 0.3, "chair": 0.3, "shelving": 0.25},
    "living_room": {"sofa": 1.0, "couch": 1.0, "tv_monitor": 0.9, "tv": 0.9, "fireplace": 0.9, "cushion": 0.7, "seating": 0.7, "chair": 0.5, "shelving": 0.45, "table": 0.4, "picture": 0.4, "plant": 0.4, "curtain": 0.35, "blinds": 0.35, "window": 0.25},
    "kitchen": {"counter": 1.0, "appliances": 0.9, "sink": 0.8, "cabinet": 0.7, "table": 0.5, "stool": 0.5, "shelving": 0.4, "chair": 0.3, "window": 0.15, "blinds": 0.15},
    "dining_room": {"table": 1.0, "chair": 0.8, "stool": 0.5, "cabinet": 0.4, "picture": 0.3, "window": 0.2, "curtain": 0.2, "blinds": 0.2, "shelving": 0.2},
    "office_room": {"chair": 0.8, "table": 0.8, "shelving": 0.6, "cabinet": 0.5, "picture": 0.3, "plant": 0.3, "window": 0.25, "blinds": 0.25},
    "gym": {"gym_equipment": 1.0, "gym equipment": 1.0, "treadmill": 1.0, "exercise machine": 1.0},
    "lounge": {"sofa": 0.9, "couch": 0.9, "seating": 0.9, "chair": 0.6, "table": 0.5, "plant": 0.4, "picture": 0.4, "tv_monitor": 0.4, "tv": 0.4},
    "laundry_room": {"clothes": 0.8, "towel": 0.7, "sink": 0.5, "cabinet": 0.4, "counter": 0.3},
    "hallway": {"stairs": 1.0, "door": 0.9, "window": 0.2, "mirror": 0.2},
    "storage_room": {"shelving": 0.8, "door": 0.3, "appliances": 0.3},
}

# Adjacency priors: if we've explored room X, what room is likely next?
# P(next_room | current_room) — from common house layouts
ROOM_ADJACENCY = {
    "bedroom": {"bathroom": 0.35, "laundry_room": 0.2, "living_room": 0.15, "lounge": 0.1, "bedroom": 0.1},
    "living_room": {"kitchen": 0.25, "dining_room": 0.2, "lounge": 0.2, "office_room": 0.1, "bedroom": 0.1, "bathroom": 0.05},
    "bathroom": {"bedroom": 0.35, "laundry_room": 0.2, "living_room": 0.15, "kitchen": 0.05},
    "kitchen": {"dining_room": 0.3, "living_room": 0.25, "laundry_room": 0.15, "lounge": 0.1, "bathroom": 0.05},
    "dining_room": {"kitchen": 0.35, "living_room": 0.25, "lounge": 0.15, "office_room": 0.05},
    "office_room": {"living_room": 0.25, "lounge": 0.25, "bedroom": 0.1, "dining_room": 0.1},
    "gym": {"lounge": 0.2, "living_room": 0.15, "bathroom": 0.1},
    "lounge": {"living_room": 0.3, "dining_room": 0.15, "office_room": 0.15, "kitchen": 0.1, "gym": 0.05},
    "laundry_room": {"kitchen": 0.25, "bedroom": 0.2, "bathroom": 0.2, "living_room": 0.1},
    "hallway": {"living_room": 0.25, "bedroom": 0.2, "bathroom": 0.15, "kitchen": 0.15, "dining_room": 0.1, "office_room": 0.1, "storage_room": 0.08},
    "storage_room": {"hallway": 0.25, "bedroom": 0.2, "kitchen": 0.18, "laundry_room": 0.12},
    "unknown": {"bedroom": 0.15, "living_room": 0.15, "bathroom": 0.15, "kitchen": 0.15, "dining_room": 0.1, "office_room": 0.1, "lounge": 0.1, "hallway": 0.08, "storage_room": 0.06, "laundry_room": 0.05, "gym": 0.05},
}

# Object co-occurrence: P(target | observed_object_nearby)
# When we see object X nearby, how much does it boost the probability of target Y?
OBJECT_COOCCURRENCE = {
    "toilet": {"sink": 0.85, "bathtub": 0.9, "shower": 0.9, "towel": 0.7, "mirror": 0.55},
    "sink": {"toilet": 0.7, "shower": 0.65, "bathtub": 0.65, "towel": 0.55, "mirror": 0.55, "counter": 0.45, "appliances": 0.55},
    "shower": {"toilet": 0.75, "sink": 0.65, "towel": 0.7, "bathtub": 0.55, "mirror": 0.45},
    "bathtub": {"toilet": 0.75, "sink": 0.65, "towel": 0.7, "shower": 0.55, "mirror": 0.45},
    "towel": {"toilet": 0.65, "sink": 0.6, "shower": 0.7, "bathtub": 0.7, "bed": 0.25},
    "bed": {"chest_of_drawers": 0.85, "clothes": 0.65, "picture": 0.45, "mirror": 0.35, "window": 0.35, "curtain": 0.35, "blinds": 0.35, "towel": 0.3},
    "chest_of_drawers": {"bed": 0.85, "clothes": 0.65, "shelving": 0.45, "mirror": 0.4, "window": 0.3, "curtain": 0.3, "blinds": 0.3, "cabinet": 0.35},
    "clothes": {"bed": 0.65, "chest_of_drawers": 0.65, "shelving": 0.55, "cabinet": 0.45, "mirror": 0.35},
    "chair": {"table": 0.7, "sofa": 0.5, "tv_monitor": 0.4, "picture": 0.3, "window": 0.25, "curtain": 0.25, "blinds": 0.25, "door": 0.15, "stairs": 0.15},
    "table": {"chair": 0.75, "stool": 0.45, "appliances": 0.35, "counter": 0.35, "shelving": 0.25, "sofa": 0.25},
    "stool": {"table": 0.55, "counter": 0.55, "chair": 0.45, "appliances": 0.35},
    "tv_monitor": {"sofa": 0.8, "chair": 0.5, "fireplace": 0.4, "seating": 0.6, "window": 0.25, "curtain": 0.25, "blinds": 0.25},
    "sofa": {"tv_monitor": 0.7, "chair": 0.5, "fireplace": 0.4, "table": 0.3, "window": 0.25, "curtain": 0.25, "blinds": 0.25, "cushion": 0.75},
    "cushion": {"sofa": 0.75, "seating": 0.65, "chair": 0.45, "window": 0.25, "curtain": 0.25, "blinds": 0.25},
    "seating": {"sofa": 0.6, "chair": 0.55, "cushion": 0.65, "tv_monitor": 0.45, "window": 0.25, "curtain": 0.25, "blinds": 0.25, "door": 0.15, "stairs": 0.15},
    "picture": {"sofa": 0.45, "bed": 0.4, "window": 0.3, "curtain": 0.3, "blinds": 0.3, "shelving": 0.25, "table": 0.25, "fireplace": 0.35, "cabinet": 0.25, "door": 0.15, "stairs": 0.15},
    "cabinet": {"counter": 0.55, "appliances": 0.55, "shelving": 0.55, "sink": 0.35, "table": 0.3, "picture": 0.25},
    "counter": {"appliances": 0.75, "sink": 0.55, "cabinet": 0.55, "stool": 0.45, "table": 0.3},
    "fireplace": {"sofa": 0.65, "tv_monitor": 0.35, "chair": 0.35, "picture": 0.35},
    "plant": {"stairs": 0.4, "chair": 0.3, "sofa": 0.3, "table": 0.3, "door": 0.25},
    "gym_equipment": {"treadmill": 0.7, "picture": 0.2},
}

# Object co-occurrence priors: P(target | room_type)
# Based on common indoor layouts
TARGET_ROOM_PRIOR = {
    "chair": {"office_room": 0.45, "dining_room": 0.4, "living_room": 0.35, "lounge": 0.3, "kitchen": 0.25, "bedroom": 0.12, "hallway": 0.1, "bathroom": 0.02},
    "table": {"dining_room": 0.55, "kitchen": 0.4, "office_room": 0.35, "living_room": 0.25, "lounge": 0.22, "hallway": 0.08, "bedroom": 0.08, "bathroom": 0.01},
    "picture": {"living_room": 0.35, "bedroom": 0.3, "office_room": 0.25, "lounge": 0.25, "hallway": 0.18, "dining_room": 0.18, "kitchen": 0.1, "bathroom": 0.06},
    "cabinet": {"kitchen": 0.4, "bedroom": 0.32, "storage_room": 0.3, "bathroom": 0.22, "office_room": 0.22, "dining_room": 0.18, "living_room": 0.16, "laundry_room": 0.14, "hallway": 0.1},
    "cushion": {"living_room": 0.75, "lounge": 0.65, "bedroom": 0.15, "dining_room": 0.05, "kitchen": 0.02, "bathroom": 0.01},
    "sofa": {"living_room": 0.75, "lounge": 0.65, "bedroom": 0.08, "dining_room": 0.04, "kitchen": 0.02, "bathroom": 0.01},
    "couch": {"living_room": 0.75, "lounge": 0.65, "bedroom": 0.08, "dining_room": 0.04, "kitchen": 0.02, "bathroom": 0.01},
    "bed": {"bedroom": 0.9, "living_room": 0.04, "lounge": 0.03, "kitchen": 0.01, "bathroom": 0.01, "dining_room": 0.01},
    "chest_of_drawers": {"bedroom": 0.85, "storage_room": 0.18, "living_room": 0.06, "laundry_room": 0.05, "bathroom": 0.02, "kitchen": 0.01},
    "drawer": {"bedroom": 0.85, "living_room": 0.06, "laundry_room": 0.05, "bathroom": 0.02, "kitchen": 0.01},
    "plant": {"living_room": 0.4, "lounge": 0.35, "office_room": 0.25, "hallway": 0.25, "dining_room": 0.18, "kitchen": 0.12, "bedroom": 0.1, "bathroom": 0.04},
    "sink": {"bathroom": 0.55, "kitchen": 0.4, "laundry_room": 0.25, "bedroom": 0.02, "living_room": 0.01},
    "toilet": {"bathroom": 0.9, "bedroom": 0.02, "kitchen": 0.02, "living_room": 0.01, "laundry_room": 0.01},
    "stool": {"kitchen": 0.42, "dining_room": 0.34, "living_room": 0.12, "lounge": 0.1, "bedroom": 0.04, "bathroom": 0.01},
    "towel": {"bathroom": 0.75, "laundry_room": 0.35, "storage_room": 0.2, "bedroom": 0.18, "kitchen": 0.08, "living_room": 0.02},
    "tv_monitor": {"living_room": 0.65, "lounge": 0.45, "bedroom": 0.2, "kitchen": 0.04, "bathroom": 0.01},
    "tv": {"living_room": 0.65, "lounge": 0.45, "bedroom": 0.2, "kitchen": 0.04, "bathroom": 0.01},
    "shower": {"bathroom": 0.9, "laundry_room": 0.04, "bedroom": 0.03, "kitchen": 0.02, "living_room": 0.01},
    "bathtub": {"bathroom": 0.9, "laundry_room": 0.03, "bedroom": 0.03, "kitchen": 0.02, "living_room": 0.01},
    "counter": {"kitchen": 0.65, "bathroom": 0.2, "laundry_room": 0.15, "dining_room": 0.1, "living_room": 0.04, "bedroom": 0.01},
    "fireplace": {"living_room": 0.75, "lounge": 0.35, "bedroom": 0.1, "dining_room": 0.06, "kitchen": 0.01},
    "gym_equipment": {"gym": 0.85, "lounge": 0.08, "living_room": 0.05, "bedroom": 0.03},
    "seating": {"living_room": 0.6, "lounge": 0.55, "dining_room": 0.22, "office_room": 0.15, "hallway": 0.08, "bedroom": 0.06, "kitchen": 0.02},
    "clothes": {"bedroom": 0.75, "laundry_room": 0.45, "storage_room": 0.28, "bathroom": 0.08, "hallway": 0.08, "living_room": 0.03, "lounge": 0.02},
}


def _clamp01(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(value, 1.0))


def _object_presence(mass, score, mass_ref=8.0):
    return _clamp01(score) * min(max(float(mass), 0.0) / mass_ref, 1.0)


def _infer_room_from_object_stats(object_stats):
    room_scores = {room: 0.03 for room in ROOM_SIGNATURES}
    for obj_name, stats in object_stats.items():
        presence = stats.get("presence", 0.0)
        if presence <= 0:
            continue
        for room, weights in ROOM_SIGNATURES.items():
            weight = weights.get(obj_name)
            if weight is not None:
                room_scores[room] += presence * weight

    ranked = sorted(room_scores.items(), key=lambda item: item[1], reverse=True)
    best_room, best_score = ranked[0]
    second_room, second_score = ranked[1] if len(ranked) > 1 else ("unknown", 0.0)
    margin = max(0.0, best_score - second_score)

    if best_score < 0.18:
        return {
            "room_type": "unknown",
            "room_confidence": 0.0,
            "best_room": best_room,
            "second_room": second_room,
            "room_margin": margin,
            "room_scores": room_scores,
        }

    evidence_conf = min(best_score / 1.0, 1.0)
    margin_conf = min(margin / 0.35, 1.0)
    room_conf = 0.65 * evidence_conf + 0.35 * margin_conf
    return {
        "room_type": best_room,
        "room_confidence": room_conf,
        "best_room": best_room,
        "second_room": second_room,
        "room_margin": margin,
        "room_scores": room_scores,
    }


def enrich_frontiers(
    full_map_pred,
    frontier_points,
    frontier_areas,
    goal_name,
    semantic_categories,
    radius=60,
):
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
    semantic_map = full_map_pred[4:]
    H, W = semantic_map.shape[1], semantic_map.shape[2]
    enriched = []

    for idx, (fy, fx) in enumerate(frontier_points):
        # Get nearby objects within radius
        y_min, y_max = max(0, fy - radius), min(H, fy + radius)
        x_min, x_max = max(0, fx - radius), min(W, fx + radius)
        patch = semantic_map[:, y_min:y_max, x_min:x_max]

        nearby_objects = []
        nearby_object_scores = {}
        nearby_object_stats = {}
        for cat_idx, cat_name in enumerate(semantic_categories):
            if cat_idx >= patch.shape[0]:
                continue
            cat_patch = patch[cat_idx]
            mass = float(cat_patch.sum().item()) if hasattr(cat_patch, "sum") else float(cat_patch.sum())
            if mass <= 2:
                continue
            score = float(cat_patch.max().item()) if hasattr(cat_patch, "max") else float(cat_patch.max())
            presence = _object_presence(mass, score)
            nearby_objects.append(cat_name)
            nearby_object_scores[cat_name] = round(_clamp01(score), 3)
            nearby_object_stats[cat_name] = {
                "mass": round(mass, 3),
                "score": round(_clamp01(score), 3),
                "presence": round(presence, 3),
            }

        # Infer room type from nearby objects
        room_inference = _infer_room_from_object_stats(nearby_object_stats)
        best_room = room_inference["room_type"]
        room_conf = room_inference["room_confidence"]

        # Target prior based on room type
        target_priors = TARGET_ROOM_PRIOR.get(goal_name, {})
        target_prior = target_priors.get(best_room, 0.05) if best_room != "unknown" else 0.1

        # Boost prior using object co-occurrence (even if room is "unknown")
        cooccurrence = OBJECT_COOCCURRENCE.get(goal_name, {})
        for obj in nearby_objects:
            if obj in cooccurrence:
                target_prior = max(target_prior, cooccurrence[obj])

        # For unknown rooms, use adjacency prior from nearby explored room-like evidence.
        if best_room == "unknown" and nearby_objects:
            big_radius = radius * 2
            by_min, by_max = max(0, fy - big_radius), min(H, fy + big_radius)
            bx_min, bx_max = max(0, fx - big_radius), min(W, fx + big_radius)
            big_patch = semantic_map[:, by_min:by_max, bx_min:bx_max]
            big_object_stats = {}
            for ci, cat_name in enumerate(semantic_categories):
                if ci >= big_patch.shape[0]:
                    continue
                cat_patch = big_patch[ci]
                mass = float(cat_patch.sum().item()) if hasattr(cat_patch, "sum") else float(cat_patch.sum())
                if mass <= 5:
                    continue
                score = float(cat_patch.max().item()) if hasattr(cat_patch, "max") else float(cat_patch.max())
                big_object_stats[cat_name] = {
                    "mass": mass,
                    "score": _clamp01(score),
                    "presence": _object_presence(mass, score),
                }

            adj_inference = _infer_room_from_object_stats(big_object_stats)
            explored_room = adj_inference["room_type"]
            if explored_room != "unknown":
                adj_priors = ROOM_ADJACENCY.get(explored_room, {})
                for adj_room, adj_prob in adj_priors.items():
                    room_target_prior = target_priors.get(adj_room, 0.05)
                    adj_boost = adj_prob * room_target_prior
                    target_prior = max(target_prior, adj_boost)
                    if adj_room == "bathroom" and goal_name == "toilet" and adj_prob > 0.2:
                        target_prior = max(target_prior, 0.5)

        frontier_area = frontier_areas[idx] if idx < len(frontier_areas) else 0
        if goal_name == "toilet" and frontier_area < 20 and best_room == "unknown":
            target_prior = max(target_prior, 0.35)

        enriched.append({
            "idx": idx,
            "centroid": (fy, fx),
            "area": frontier_area,
            "nearby_objects": nearby_objects,
            "nearby_object_scores": nearby_object_scores,
            "nearby_object_stats": nearby_object_stats,
            "room_type": best_room if room_conf > 0 else "unknown",
            "room_confidence": round(room_conf, 2),
            "second_room": room_inference["second_room"],
            "room_margin": round(room_inference["room_margin"], 3),
            "room_scores": {room: round(score, 3) for room, score in room_inference["room_scores"].items()},
            "target_prior": round(target_prior, 2),
        })

    return enriched


def Objects_Extract(full_map_pred, semantic_categories):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(min(len(semantic_map), len(semantic_categories))):
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
                Object_list[semantic_categories[i]] = Single_object_list

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


def request_brain_response(args, messages, max_tokens=100):
    """Return a frontier assignment from the configured MindNav brain backend."""
    backend = getattr(args, "brain_backend", "local")
    if backend == "siliconflow":
        import openai as _oai

        api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for --brain_backend=siliconflow")

        base_url = getattr(args, "brain_base_url", None) or "https://api.siliconflow.cn/v1"
        client = _oai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=args.brain_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content.strip(), args.brain_model

    if backend == "deepseek":
        import urllib.error
        import urllib.request

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for --brain_backend=deepseek")

        base_url = (
            getattr(args, "brain_base_url", None)
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        model = getattr(args, "brain_model", None) or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        if model == "Pro/MiniMaxAI/MiniMax-M2.5":
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if getattr(args, "deepseek_thinking", os.environ.get("DEEPSEEK_THINKING", "disabled")) == "disabled":
            payload["thinking"] = {"type": "disabled"}

        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API request failed: HTTP {exc.code} {detail[:500]}") from exc

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content is None:
            content = ""
        return content.strip(), model

    response = chat_completion_create(
        model=gpt_name[min(args.gpt_type, len(gpt_name) - 1)],
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"].strip(), args.llm_path or gpt_name[min(args.gpt_type, len(gpt_name) - 1)]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_jsonl(path, record):
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")


def derived_jsonl_path(base_path, suffix):
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".jsonl"
    return root + suffix + ext


def init_analysis_jsonls(args):
    if args.jsonl_log:
        if not args.decision_jsonl:
            args.decision_jsonl = derived_jsonl_path(args.jsonl_log, ".decisions")
        if not args.stop_diag_jsonl:
            args.stop_diag_jsonl = derived_jsonl_path(args.jsonl_log, ".stop_diag")

    for attr in ("jsonl_log", "decision_jsonl", "stop_diag_jsonl"):
        path = getattr(args, attr, None)
        if not path:
            continue
        path_dir = os.path.dirname(path)
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)
        if attr == "jsonl_log" and args.append_jsonl and os.path.exists(path):
            with open(path) as fh:
                args.jsonl_episode_offset = sum(1 for line in fh if line.strip())
        elif args.append_jsonl and os.path.exists(path):
            pass
        else:
            open(path, "w").close()
            if attr == "jsonl_log":
                args.jsonl_episode_offset = int(args.start_episode_index or 0)


def get_episode_metadata(env):
    episode = getattr(env, "current_episode", None)
    if episode is None:
        episode = getattr(getattr(env, "habitat_env", None), "current_episode", None)
    return {
        "episode_id": getattr(episode, "episode_id", None),
        "scene_id": getattr(episode, "scene_id", None),
    }


def summarize_frontiers_for_log(frontiers):
    out = []
    for ef in frontiers:
        out.append({
            "idx": int(ef.get("idx", -1)),
            "centroid": list(ef.get("centroid", [])),
            "area": float(ef.get("area", 0.0)),
            "room_type": ef.get("room_type", "unknown"),
            "room_confidence": float(ef.get("room_confidence", 0.0)),
            "target_prior": float(ef.get("target_prior", 0.0)),
            "nearby_objects": list(ef.get("nearby_objects", [])),
        })
    return out


def robot_poses_for_log(pose_pred):
    return [
        {"x": float(p[0]), "y": float(p[1]), "theta": float(p[2])}
        for p in pose_pred
    ]


def target_channel_stats(map_tensor, target_name, args):
    empty = {
        "target_channel_mass": 0.0,
        "target_channel_max_score": 0.0,
        "target_largest_cc_area": 0,
        "target_num_components": 0,
        "target_centroid": None,
    }
    if map_tensor is None or not target_name:
        return empty
    ch = args.category_to_channel.get(target_name)
    if ch is None or ch + 4 >= map_tensor.shape[0]:
        return empty
    arr = map_tensor[ch + 4].detach().float().cpu().numpy()
    mask = arr > float(getattr(args, "target_map_score_thr", 0.1))
    if not np.any(mask):
        return {
            "target_channel_mass": float(arr.sum()),
            "target_channel_max_score": float(arr.max()) if arr.size else 0.0,
            "target_largest_cc_area": 0,
            "target_num_components": 0,
            "target_centroid": None,
        }
    labels, num = measure.label(mask.astype(np.uint8), connectivity=2, return_num=True)
    areas = np.bincount(labels.ravel())[1:] if num > 0 else np.array([])
    largest_area = int(areas.max()) if areas.size else 0
    ys, xs = np.where(mask)
    return {
        "target_channel_mass": float(arr.sum()),
        "target_channel_max_score": float(arr.max()) if arr.size else 0.0,
        "target_largest_cc_area": largest_area,
        "target_num_components": int(num),
        "target_centroid": [float(ys.mean()), float(xs.mean())],
    }


def agent_target_diagnostics(agent, args):
    out = {
        "target_stop_mode": getattr(args, "target_stop_mode", "enforce"),
        "found_goal_reason": getattr(agent, "last_found_goal_reason", None),
    }
    out.update(dict(getattr(agent, "last_target_map_evidence", {}) or {}))
    out.update(dict(getattr(agent, "last_fresh_target_evidence", {}) or {}))
    out.update(dict(getattr(agent, "last_target_confirmation", {}) or {}))
    return out


def classify_end_reason(metrics, final_action, steps, args):
    success = float(metrics.get("success", 0.0)) >= 0.5
    stop_called = any(int(a) == 0 for a in (final_action or []))
    if success:
        return "success_stop" if stop_called else "success"
    if stop_called:
        return "false_stop"
    if steps >= int(args.max_episode_length):
        return "timeout"
    return "episode_over"


def write_decision_jsonl(args, record):
    write_jsonl(args.decision_jsonl, record)


def write_stop_diag_jsonl(args, record):
    write_jsonl(args.stop_diag_jsonl, record)


def write_episode_jsonl(args, episode_idx, metrics, extra=None):
    if not args.jsonl_log:
        return
    episode_offset = int(getattr(args, "jsonl_episode_offset", 0))
    record = {
        "method": args.method_name or "mindnav",
        "episode": episode_offset + episode_idx,
        "success": float(metrics.get("success", 0.0)),
        "spl": float(metrics.get("spl", metrics.get("SPL", 0.0))),
    }
    if extra:
        record.update(extra)
    write_jsonl(args.jsonl_log, record)


AGENT_STATE_DEFAULTS = {
    "guard": "1",
    "arrival_radius_px": 12.0,
    "match_radius_px": 25.0,
    "progress_eps_px": 2.0,
    "no_progress_steps": 20,
    "switch_margin": 0.25,
    "max_active_steps": 75,
}


def agent_state_guard_enabled():
    return bool(int(os.environ.get("AGENT_STATE_GUARD", AGENT_STATE_DEFAULTS["guard"])))


def agent_state_cfg_float(name):
    return float(os.environ.get(f"AGENT_STATE_{name}", AGENT_STATE_DEFAULTS[name.lower()]))


def agent_state_cfg_int(name):
    return int(os.environ.get(f"AGENT_STATE_{name}", AGENT_STATE_DEFAULTS[name.lower()]))


def init_agent_run_states(num_agents):
    return [
        {
            "active": False,
            "frontier_idx": None,
            "goal_point": None,
            "assigned_step": -1,
            "last_dist": None,
            "best_dist": float("inf"),
            "no_progress_steps": 0,
            "status": "idle",
            "target_prior": 0.0,
            "last_reason": "episode_reset",
        }
        for _ in range(num_agents)
    ]


def grid_position_from_pose(args, start_x, start_y):
    return [
        int(start_y * 100.0 / args.map_resolution),
        int(start_x * 100.0 / args.map_resolution),
    ]


def grid_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def frontier_prior(enriched_frontiers, frontier_idx, default=0.0):
    for ef in enriched_frontiers:
        if int(ef.get("idx", -1)) == int(frontier_idx):
            return float(ef.get("target_prior", default))
    return float(default)


def match_goal_to_frontier(goal_point, target_point_map, match_radius_px):
    if goal_point is None or not target_point_map:
        return None, float("inf")
    best_idx = None
    best_dist = float("inf")
    for idx, point in enumerate(target_point_map):
        dist = grid_distance(goal_point, point)
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    if best_dist <= match_radius_px:
        return best_idx, best_dist
    return None, best_dist


def update_agent_run_state(state, grid_pos, agent_obj, step):
    if not state.get("active"):
        return

    goal_point = state.get("goal_point")
    dist = grid_distance(grid_pos, goal_point)
    state["last_dist"] = dist

    arrival_radius = agent_state_cfg_float("ARRIVAL_RADIUS_PX")
    progress_eps = agent_state_cfg_float("PROGRESS_EPS_PX")
    no_progress_limit = agent_state_cfg_int("NO_PROGRESS_STEPS")
    max_active_steps = agent_state_cfg_int("MAX_ACTIVE_STEPS")

    if dist <= arrival_radius:
        state["active"] = False
        state["status"] = "arrived"
        state["last_reason"] = f"arrived_dist={dist:.1f}"
        return

    if dist < float(state.get("best_dist", float("inf"))) - progress_eps:
        state["best_dist"] = dist
        state["no_progress_steps"] = 0
        state["status"] = "moving"
        state["last_reason"] = "progress"
    else:
        state["no_progress_steps"] = int(state.get("no_progress_steps", 0)) + 1
        state["status"] = "moving"
        state["last_reason"] = "no_progress"

    assigned_for = step - int(state.get("assigned_step", step))
    if state["no_progress_steps"] >= no_progress_limit:
        state["active"] = False
        state["status"] = "stuck"
        state["last_reason"] = f"no_progress_steps={state['no_progress_steps']}"
    elif assigned_for >= max_active_steps:
        state["active"] = False
        state["status"] = "stuck"
        state["last_reason"] = f"max_active_steps={assigned_for}"
    elif getattr(agent_obj, "replan_count", 0) >= max(5, no_progress_limit // 2):
        state["active"] = False
        state["status"] = "stuck"
        state["last_reason"] = f"replan_count={agent_obj.replan_count}"


def should_keep_active_goal(state, proposed_idx, target_point_map, enriched_frontiers):
    if not agent_state_guard_enabled():
        return False, None, "guard_disabled"
    if not state.get("active"):
        return False, None, f"state_{state.get('status', 'idle')}"

    match_radius = agent_state_cfg_float("MATCH_RADIUS_PX")
    matched_idx, match_dist = match_goal_to_frontier(
        state.get("goal_point"),
        target_point_map,
        match_radius,
    )
    if matched_idx is None:
        state["active"] = False
        state["status"] = "lost"
        state["last_reason"] = f"frontier_lost_dist={match_dist:.1f}"
        return False, None, state["last_reason"]

    old_prior = max(
        float(state.get("target_prior", 0.0)),
        frontier_prior(enriched_frontiers, matched_idx, 0.0),
    )
    new_prior = frontier_prior(enriched_frontiers, proposed_idx, 0.0)
    switch_margin = agent_state_cfg_float("SWITCH_MARGIN")
    if new_prior >= old_prior + switch_margin:
        return False, matched_idx, f"new_prior_better={new_prior:.2f}>{old_prior:.2f}"

    return True, matched_idx, f"keep_active_goal matched=frontier_{matched_idx}"


def assign_agent_goal_state(state, frontier_idx, goal_point, target_prior, step, grid_pos, reason):
    dist = grid_distance(grid_pos, goal_point)
    state.update({
        "active": True,
        "frontier_idx": int(frontier_idx),
        "goal_point": [int(goal_point[0]), int(goal_point[1])],
        "assigned_step": int(step),
        "last_dist": dist,
        "best_dist": dist,
        "no_progress_steps": 0,
        "status": "moving",
        "target_prior": float(target_prior),
        "last_reason": reason,
    })


def repair_duplicate_frontiers(goal_frontiers, kept_flags, enriched_frontiers, num_agents):
    if num_agents < 2 or len(set(goal_frontiers.values())) == len(goal_frontiers):
        return goal_frontiers, [], set()
    if len(enriched_frontiers) < 2:
        return goal_frontiers, [], set()

    logs = []
    moved = set()
    robots = [f"robot_{i}" for i in range(num_agents) if f"robot_{i}" in goal_frontiers]
    used = {}
    for robot in robots:
        idx = int(goal_frontiers[robot])
        if idx not in used:
            used[idx] = robot
            continue

        previous = used[idx]
        if kept_flags.get(robot) and not kept_flags.get(previous):
            move_robot = previous
        else:
            move_robot = robot

        occupied = {int(v) for r, v in goal_frontiers.items() if r != move_robot}
        candidates = [
            ef for ef in enriched_frontiers
            if int(ef.get("idx", -1)) not in occupied
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda ef: float(ef.get("target_prior", 0.0)))
        old_idx = goal_frontiers[move_robot]
        goal_frontiers[move_robot] = int(best["idx"])
        moved.add(move_robot)
        logs.append(
            f"{move_robot}:duplicate frontier_{old_idx}->frontier_{goal_frontiers[move_robot]}"
        )
    return goal_frontiers, logs, moved


def mark_agent_goals_lost(agent_run_states, reason):
    for state in agent_run_states:
        if state.get("active"):
            state["active"] = False
            state["status"] = "lost"
            state["last_reason"] = reason


def finalize_frontier_assignments(goal_frontiers, agent_run_states, agent_grid_positions,
                                  target_point_map, enriched_frontiers, frontiers_dict,
                                  last_decision, step, num_agents, method_name):
    final_goal_frontiers = {}
    kept_flags = {}
    logs = []
    target_found = method_name == "mindnav_kg_target_found"

    for i in range(num_agents):
        robot_key = f"robot_{i}"
        proposed_idx = max(0, min(int(goal_frontiers[robot_key]), len(target_point_map) - 1))
        keep = False
        matched_idx = None
        reason = "switch_new_goal"
        if not target_found:
            keep, matched_idx, reason = should_keep_active_goal(
                agent_run_states[i],
                proposed_idx,
                target_point_map,
                enriched_frontiers,
            )
        if keep:
            final_goal_frontiers[robot_key] = int(matched_idx)
            kept_flags[robot_key] = True
        else:
            final_goal_frontiers[robot_key] = int(proposed_idx)
            kept_flags[robot_key] = False
        logs.append(
            f"{robot_key}:raw=frontier_{proposed_idx} final=frontier_{final_goal_frontiers[robot_key]} "
            f"keep={kept_flags[robot_key]} reason={reason}"
        )

    final_goal_frontiers, duplicate_logs, moved_robots = repair_duplicate_frontiers(
        final_goal_frontiers,
        kept_flags,
        enriched_frontiers,
        num_agents,
    )
    logs.extend(duplicate_logs)
    for robot_key in moved_robots:
        kept_flags[robot_key] = False

    final_goal_points = []
    last_decision.clear()
    for i in range(num_agents):
        robot_key = f"robot_{i}"
        frontier_idx = max(0, min(int(final_goal_frontiers[robot_key]), len(target_point_map) - 1))
        point = target_point_map[frontier_idx]
        prior = frontier_prior(enriched_frontiers, frontier_idx, 0.0)
        final_goal_points.append(point)
        last_decision.append(frontiers_dict.get(f"frontier_{frontier_idx}", ""))

        if kept_flags.get(robot_key) and agent_run_states[i].get("active"):
            agent_run_states[i]["frontier_idx"] = int(frontier_idx)
            agent_run_states[i]["goal_point"] = [int(point[0]), int(point[1])]
            agent_run_states[i]["target_prior"] = max(
                float(agent_run_states[i].get("target_prior", 0.0)),
                prior,
            )
            agent_run_states[i]["last_reason"] = "keep_active_goal"
        else:
            assign_agent_goal_state(
                agent_run_states[i],
                frontier_idx,
                point,
                prior,
                step,
                agent_grid_positions[i],
                "target_found" if target_found else "switch_new_goal",
            )

    return final_goal_points, final_goal_frontiers, kept_flags, logs


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


@habitat.registry.register_task_action
class TurnLeftAction_S(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.TURN_LEFT_S)


@habitat.registry.register_task_action
class TurnRightAction_S(SimulatorTaskAction):
    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.TURN_RIGHT_S)

def main():
    args = get_args()
    mindnav_config = load_mindnav_config(args.mindnav_config)
    kg_cfg = mindnav_config.get("kg", {}) if isinstance(mindnav_config.get("kg", {}), dict) else {}
    brain_cfg = mindnav_config.get("brain", {}) if isinstance(mindnav_config.get("brain", {}), dict) else {}

    if args.llm_path is None and brain_cfg.get("llm_path"):
        args.llm_path = brain_cfg["llm_path"]
    if "max_tokens" in brain_cfg and "BRAIN_MAX_TOKENS" not in os.environ:
        args.brain_max_tokens = int(cfg_number(brain_cfg, "max_tokens", args.brain_max_tokens, int))
    if "priority_target_tau" in kg_cfg and "MINDNAV_TARGET_TAU" not in os.environ:
        args.mindnav_target_tau = cfg_number(kg_cfg, "priority_target_tau", args.mindnav_target_tau, float)
    if "max_steps" in mindnav_config and "MAX_EPISODE_LENGTH" not in os.environ:
        args.max_episode_length = int(cfg_number(mindnav_config, "max_steps", args.max_episode_length, int))
    if "global_plan_every" in mindnav_config and "NUM_LOCAL_STEPS" not in os.environ:
        args.num_local_steps = int(cfg_number(mindnav_config, "global_plan_every", args.num_local_steps, int))
    frontier_enrichment_radius = int(cfg_number(kg_cfg, "frontier_radius_px", kg_cfg.get("near_px", 30), int))
    if mindnav_config:
        print(f"Loaded MindNav config: {mindnav_config.get('_config_path', args.mindnav_config)}")

    # Load local LLM on llm_gpu_id when provided; keep it separate from sim/semantic GPUs.
    if args.brain_backend == "local" and args.mindnav_mode != "heuristic":
        model_idx = min(args.gpt_type, len(LOCAL_MODEL_PATHS) - 1)
        model_path = args.llm_path or LOCAL_MODEL_PATHS[model_idx]
        model_type = "vl" if "VL" in model_path else "text"
        llm_gpu_id = args.llm_gpu_id if args.llm_gpu_id >= 0 else args.sem_gpu_id
        vlm_device = f"cuda:{llm_gpu_id}"
        load_model(model_path, device=vlm_device, model_type=model_type)
    elif args.mindnav_mode == "heuristic":
        print("Using MindNav heuristic brain: no LLM will be loaded or called")
    else:
        print(f"Using remote MindNav brain: {args.brain_model}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()

    # Apply --split argument to dataset config
    config_env.DATASET.SPLIT = args.split
    config_env.DATASET.DATA_PATH = config_env.DATASET.DATA_PATH.replace("{split}", args.split)
    try:
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id
    except Exception:
        pass
    try:
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
    except Exception:
        pass

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


    env_cls = GTMultiAgentEnv if args.use_gtsem else PredMultiAgentEnv
    if args.use_gtsem:
        print("Using Habitat GT semantic observations")
    env = env_cls(config_env=config_env)

    skipped_episodes = 0
    if args.start_episode_index and args.start_episode_index > 0:
        for _ in range(min(args.start_episode_index, env.number_of_episodes)):
            env.current_episode = next(env.episode_iterator)
            skipped_episodes += 1
        print(f"Skipped {skipped_episodes} Habitat iterator episodes before evaluation")

    num_episodes = env.number_of_episodes - skipped_episodes
    if args.max_episodes and args.max_episodes > 0:
        num_episodes = min(num_episodes, args.max_episodes)

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS
    agent = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, i))

    kg = KnowledgeGraph(certainty_cap=cfg_number(kg_cfg, "certainty_cap", 0.99, float))
    kg_updater = KGUpdater(
        kg,
        room_grid_scale=cfg_number(kg_cfg, "room_grid_scale", 50, int),
        merge_radius_px=cfg_number(kg_cfg, "merge_radius_px", 30, float),
        object_merge_enabled=cfg_bool(kg_cfg, "object_merge_enabled", True),
        object_merge_radius_px=cfg_number(kg_cfg, "object_merge_radius_px", 60, float),
        next_to_px=cfg_number(kg_cfg, "next_to_px", 20, float),
        near_px=cfg_number(kg_cfg, "near_px", 60, float),
        room_connect_px=cfg_number(kg_cfg, "room_connect_px", 150, float),
        max_room_connections=cfg_number(kg_cfg, "max_room_connections", 3, int),
        create_pseudo_doors=cfg_bool(kg_cfg, "create_pseudo_doors", False),
        suggests_as_property=cfg_bool(kg_cfg, "suggests_as_property", True),
        certainty_cap=cfg_number(kg_cfg, "certainty_cap", 0.99, float),
    )

    def call_kg_model(messages, max_tokens):
        response_text, _ = request_brain_response(args, messages, max_tokens=max_tokens)
        return response_text

    kg_brain = KGToolCallingBrain(
        call_model=call_kg_model,
        num_agents=num_agents,
        max_tokens=args.brain_max_tokens,
        target_tau=args.mindnav_target_tau,
    )
    heuristic_brain = MindNavHeuristicBrain(
        num_agents=num_agents,
        target_tau=args.mindnav_target_tau,
    )
    decision_history = []
    kg_trace_logger = None
    if args.kg_trace_dir:
        kg_trace_logger = KGTraceLogger(
            args.kg_trace_dir,
            run_name=args.exp_name,
            make_plots=bool(args.kg_trace_plots),
        )


    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    init_analysis_jsonls(args)

    logging.basicConfig(
        filename=log_dir + 'output.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    logging.info(args)
    # ------------------------------------------------------------------


    device = torch.device("cuda:0" if args.cuda else "cpu")

    agg_metrics: Dict = defaultdict(float)

    count_episodes = 0
    count_step = 0
    goal_points = []
    log_start = time.time()
    last_decision = []
    total_usage = []

    while count_episodes < num_episodes:
        observations = env.reset()
        episode_number = int(args.start_episode_index or 0) + count_episodes + 1
        episode_meta = get_episode_metadata(env)
        for i in range(num_agents):
            agent[i].reset()
        kg.reset()
        decision_history.clear()
        last_decision.clear()
        goal_points.clear()
        agent_run_states = init_agent_run_states(num_agents)
        prev_found_goal = [False for _ in range(num_agents)]
        target_seen_ever = False
        first_target_seen_step = None
        min_distance_to_goal = float("inf")
        final_action = None
        last_full_target_stats = None

        while not env.episode_over:
            action = [0 for _ in range(num_agents)]
            full_map = []
            visited_vis = []
            pose_pred = []
            agent_grid_positions = []
            start = time.time()
            for i in range(num_agents):
                agent[i].mapping(observations[i])
                full_map.append(agent[i].local_map)
                visited_vis.append(agent[i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                agent_grid_positions.append(grid_position_from_pose(args, start_x, start_y))

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[i].visited_vis.shape[0],
                    (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)
                
            full_map2 = torch.stack(full_map, 0)

            full_map_pred, _ = torch.max(full_map2, 0)
            for i in range(num_agents):
                update_agent_run_state(
                    agent_run_states[i],
                    agent_grid_positions[i],
                    agent[i],
                    agent[0].l_step,
                )

            # mapping_end = time.time()
            # mapping_time = mapping_end - start
            # print('mapping_time: %.3f秒'%mapping_time)

            if agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0:
                goal_points.clear()

                Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred)

                if len(target_point_map) > 0:
                    object_list = Objects_Extract(full_map_pred, args.semantic_categories)

                    # Enrich frontiers with room-type + nearby objects + target prior
                    enriched = enrich_frontiers(
                        full_map_pred,
                        target_point_map,
                        Frontier_list,
                        agent[0].goal_name,
                        args.semantic_categories,
                        radius=frontier_enrichment_radius,
                    )

                    Frontiers_dict = {
                        "frontier_" + str(i): f"<centroid: {target_point_map[i][0], target_point_map[i][1]}, number: {Frontier_list[i]}>"
                        for i in range(len(target_point_map))
                    }
                    retries = 3
                    vlm_success = False

                    while retries > 0:
                        try:
                            if args.mindnav_mode in ("kg", "heuristic"):
                                kg_updater.update(
                                    enriched,
                                    object_list,
                                    pose_pred,
                                    Wall_list,
                                    full_map_pred,
                                    agent[0].goal_name,
                                    semantic_categories=args.semantic_categories,
                                )

                            brain_trace = {}
                            if args.mindnav_mode == "kg":
                                goal_frontiers, tools_called, method_name = kg_brain.decide(
                                    kg,
                                    agent[0].goal_name,
                                    enriched,
                                    pose_pred,
                                    agent[0].l_step,
                                    args.max_episode_length,
                                    decision_history,
                                )
                                brain_trace = dict(getattr(kg_brain, "last_trace", {}) or {})
                                print(f"MindNav KG brain: {method_name}; tools={tools_called}")
                            elif args.mindnav_mode == "heuristic":
                                goal_frontiers, tools_called, method_name = heuristic_brain.decide(
                                    kg,
                                    agent[0].goal_name,
                                    enriched,
                                    pose_pred,
                                    agent[0].l_step,
                                    args.max_episode_length,
                                    decision_history,
                                )
                                brain_trace = {"mode": "heuristic", "goal_frontiers": dict(goal_frontiers)}
                                print(f"MindNav heuristic brain: {method_name}; tools={tools_called}")
                            else:
                                User_prompt, _ = form_prompt_for_chatgpt(
                                    agent[0].goal_name,
                                    pose_pred,
                                    object_list,
                                    Wall_list,
                                    Frontier_list,
                                    last_decision,
                                    target_point_map,
                                    enriched_frontiers=enriched,
                                )
                                message_list = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": User_prompt},
                                ]
                                response_message, brain_name = request_brain_response(args, message_list)
                                print(f"Brain ({brain_name}) response:")
                                print(response_message)
                                goal_frontiers = parse_answer(response_message)
                                tools_called = ["semantic_prompt"]
                                method_name = "mindnav_semantic_prompt"
                                brain_trace = {
                                    "mode": "semantic_prompt",
                                    "decision_prompt": User_prompt,
                                    "decision_raw": response_message,
                                    "goal_frontiers": dict(goal_frontiers),
                                }
                            brain_goal_frontiers = dict(goal_frontiers)

                            # Verify we got assignments for all robots
                            for i in range(num_agents):
                                robot_key = "robot_" + str(i)
                                if robot_key not in goal_frontiers:
                                    raise ValueError(f"Missing frontier assignment for {robot_key}")
                                goal_frontiers[robot_key] = max(
                                    0,
                                    min(int(goal_frontiers[robot_key]), len(target_point_map) - 1),
                                )

                            # Preserve spatial diversity if the model collapses to one frontier.
                            if num_agents >= 2 and len(target_point_map) >= 2:
                                if goal_frontiers.get("robot_0") == goal_frontiers.get("robot_1"):
                                    chosen = goal_frontiers["robot_0"]
                                    best_alt = None
                                    best_alt_prior = -1
                                    for ef in enriched:
                                        if ef["idx"] != chosen and ef["target_prior"] > best_alt_prior:
                                            best_alt = ef["idx"]
                                            best_alt_prior = ef["target_prior"]
                                    if best_alt is not None:
                                        goal_frontiers["robot_1"] = best_alt
                                        print(f"  [FIX] Redirected robot_1 from frontier_{chosen} to frontier_{best_alt}")

                            raw_goal_frontiers = dict(goal_frontiers)
                            goal_points, goal_frontiers, kept_flags, agent_state_logs = finalize_frontier_assignments(
                                goal_frontiers,
                                agent_run_states,
                                agent_grid_positions,
                                target_point_map,
                                enriched,
                                Frontiers_dict,
                                last_decision,
                                agent[0].l_step,
                                num_agents,
                                method_name,
                            )

                            chosen_0 = goal_frontiers.get("robot_0", 0)
                            chosen_1 = goal_frontiers.get("robot_1", 0)
                            decision_history.append({
                                "step": agent[0].l_step,
                                "assignments": dict(goal_frontiers),
                                "raw_assignments": raw_goal_frontiers,
                                "tools": tools_called,
                                "method": method_name,
                            })
                            write_decision_jsonl(args, {
                                "method": args.method_name or "mindnav",
                                "episode": episode_number,
                                "step": agent[0].l_step,
                                "goal_name": agent[0].goal_name,
                                "robot_poses": robot_poses_for_log(pose_pred),
                                "frontiers": summarize_frontiers_for_log(enriched),
                                "query_prompt": brain_trace.get("query_prompt", ""),
                                "query_raw": brain_trace.get("query_raw", ""),
                                "query_tool_calls": brain_trace.get("query_tool_calls", []),
                                "query_results": brain_trace.get("query_results", []),
                                "decision_prompt": brain_trace.get("decision_prompt", ""),
                                "decision_raw": brain_trace.get("decision_raw", ""),
                                "decision_tool_calls": brain_trace.get("decision_tool_calls", []),
                                "decision_results": brain_trace.get("decision_results", []),
                                "llm_probabilities": brain_trace.get("llm_probabilities", {}),
                                "brain_assignments": brain_goal_frontiers,
                                "raw_assignments": raw_goal_frontiers,
                                "final_assignments": dict(goal_frontiers),
                                "tools_called": tools_called,
                                "decision_method": method_name,
                            })
                            if kg_trace_logger is not None and args.mindnav_mode in ("kg", "heuristic"):
                                kg_trace_logger.log_step(
                                    episode_idx=episode_number,
                                    step_idx=agent[0].l_step,
                                    goal_name=agent[0].goal_name,
                                    kg=kg,
                                    robot_poses=pose_pred,
                                    frontiers=enriched,
                                    assignments=goal_frontiers,
                                    tools=tools_called,
                                    method_name=method_name,
                                    full_map_pred=full_map_pred,
                                )
                            print(f"  [MINDNAV] step={agent[0].l_step}, goal={agent[0].goal_name}, "
                                  f"chose=({chosen_0},{chosen_1}), same={'Y' if chosen_0==chosen_1 else 'N'}, "
                                  f"kg_nodes={len(kg.nodes)}, kg_edges={len(kg.edges)}, "
                                  f"rooms={[e['room_type'] for e in enriched]}, "
                                  f"priors={[e['target_prior'] for e in enriched]}")
                            for state_log in agent_state_logs:
                                print(f"  [AGENT_STATE] step={agent[0].l_step} {state_log}")

                            vlm_success = True
                            total_usage.append(0)
                            break
                        except Exception as e:
                            print(f"Brain error: {e}")
                            print('Retrying...')
                            retries -= 1
                            time.sleep(1)

                    # Fallback: random frontier assignment if VLM failed
                    if not vlm_success:
                        print("VLM failed all retries, using random frontier assignment")
                        n_frontiers = len(target_point_map)
                        random_order = np.random.permutation(n_frontiers).tolist()
                        goal_frontiers = {
                            f"robot_{i}": random_order[i % n_frontiers]
                            for i in range(num_agents)
                        }
                        goal_points, goal_frontiers, _, agent_state_logs = finalize_frontier_assignments(
                            goal_frontiers,
                            agent_run_states,
                            agent_grid_positions,
                            target_point_map,
                            enriched,
                            Frontiers_dict,
                            last_decision,
                            agent[0].l_step,
                            num_agents,
                            "mindnav_random_fallback",
                        )
                        for state_log in agent_state_logs:
                            print(f"  [AGENT_STATE] step={agent[0].l_step} {state_log}")
                else:
                    mark_agent_goals_lost(agent_run_states, "no_frontiers")
                    for i in range(num_agents):
                        actions = np.random.rand(1, 2).squeeze()*(target_edge_map.shape[0] - 1)

                        goal_points.append([int(actions[0]), int(actions[1])])

            # start_act = time.time()
            for i in range(num_agents):
                action[i] = agent[i].act(goal_points[i])
            # act_end = time.time()
            # act_time = act_end - start_act
            # print('act_time: %.3f秒'%act_time)
            local_target_stats = [
                target_channel_stats(agent[i].local_map, agent[0].goal_name, args)
                for i in range(num_agents)
            ]
            last_full_target_stats = target_channel_stats(full_map_pred, agent[0].goal_name, args)
            for i in range(num_agents):
                found_goal_now = bool(getattr(agent[i], "last_found_goal", False))
                if found_goal_now and not prev_found_goal[i]:
                    if first_target_seen_step is None:
                        first_target_seen_step = agent[0].l_step
                    target_seen_ever = True
                    write_stop_diag_jsonl(args, {
                        "method": args.method_name or "mindnav",
                        "episode": episode_number,
                        "step": agent[0].l_step,
                        "goal_name": agent[0].goal_name,
                        "agent_id": i,
                        "event": "target_first_seen",
                        "found_goal": found_goal_now,
                        "planner_stop": bool(getattr(agent[i], "last_planner_stop", False)),
                        "action": int(action[i]),
                        "distance_to_goal": None,
                        "success": None,
                        **local_target_stats[i],
                        **agent_target_diagnostics(agent[i], args),
                    })
                prev_found_goal[i] = found_goal_now

            action = sanitize_navigation_actions(args, action)
            final_action = list(action)

            observations = env.step(action)
            try:
                step_metrics = env.get_metrics()
            except Exception:
                step_metrics = {}
            if "distance_to_goal" in step_metrics:
                min_distance_to_goal = min(
                    min_distance_to_goal,
                    float(step_metrics.get("distance_to_goal", float("inf"))),
                )
            stop_agent_ids = [i for i, act in enumerate(action) if int(act) == 0]
            for i in stop_agent_ids:
                write_stop_diag_jsonl(args, {
                    "method": args.method_name or "mindnav",
                    "episode": episode_number,
                    "step": agent[0].l_step,
                    "goal_name": agent[0].goal_name,
                    "agent_id": i,
                    "event": "stop_action",
                    "found_goal": bool(getattr(agent[i], "last_found_goal", False)),
                    "planner_stop": bool(getattr(agent[i], "last_planner_stop", False)),
                    "action": int(action[i]),
                    "distance_to_goal": float(step_metrics.get("distance_to_goal", -1.0)),
                    "success": float(step_metrics.get("success", 0.0)),
                    **local_target_stats[i],
                    **agent_target_diagnostics(agent[i], args),
                })
            if env.episode_over:
                try:
                    end_metrics = env.get_metrics()
                except Exception:
                    end_metrics = {}
                diag_agent_id = stop_agent_ids[0] if stop_agent_ids else 0
                write_stop_diag_jsonl(args, {
                    "method": args.method_name or "mindnav",
                    "episode": episode_number,
                    "step": agent[0].l_step,
                    "goal_name": agent[0].goal_name,
                    "agent_id": stop_agent_ids[0] if stop_agent_ids else -1,
                    "event": "episode_end",
                    "found_goal": any(bool(getattr(a, "last_found_goal", False)) for a in agent),
                    "planner_stop": any(bool(getattr(a, "last_planner_stop", False)) for a in agent),
                    "action": list(action),
                    "distance_to_goal": float(end_metrics.get("distance_to_goal", -1.0)),
                    "success": float(end_metrics.get("success", 0.0)),
                    **(last_full_target_stats or {}),
                    **agent_target_diagnostics(agent[diag_agent_id], args),
                })
                print(
                    "[EPISODE_END] step={} action={} distance_to_goal={:.3f} success={:.3f} spl={:.3f}".format(
                        agent[0].l_step,
                        action,
                        float(end_metrics.get("distance_to_goal", -1.0)),
                        float(end_metrics.get("success", 0.0)),
                        float(end_metrics.get("spl", end_metrics.get("SPL", 0.0))),
                    )
                )
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
        final_target_stats = last_full_target_stats or target_channel_stats(full_map_pred, agent[0].goal_name, args)
        final_diag_agent_id = 0
        if final_action:
            stopped_agents = [idx for idx, act in enumerate(final_action) if int(act) == 0]
            if stopped_agents:
                final_diag_agent_id = stopped_agents[0]
        final_agent_diag = agent_target_diagnostics(agent[final_diag_agent_id], args)
        episode_extra = {
            **episode_meta,
            "dataset": args.dataset,
            "split": args.split,
            "task_config": args.task_config,
            "use_gtsem": int(args.use_gtsem),
            "semantic_boost_backend": args.semantic_boost_backend,
            "goal_id": int(agent[0].goal_id) if agent[0].goal_id is not None else None,
            "goal_name": agent[0].goal_name,
            "steps": int(agent[0].l_step),
            "distance_to_goal": float(metrics.get("distance_to_goal", -1.0)),
            "end_reason": classify_end_reason(metrics, final_action, agent[0].l_step, args),
            "final_action": final_action,
            "target_seen_ever": bool(target_seen_ever),
            "first_target_seen_step": first_target_seen_step,
            "min_distance_to_goal": None if min_distance_to_goal == float("inf") else float(min_distance_to_goal),
            "target_stop_mode": getattr(args, "target_stop_mode", "enforce"),
            **final_target_stats,
            **final_agent_diag,
        }
        write_episode_jsonl(args, count_episodes, metrics, extra=episode_extra)
        if kg_trace_logger is not None and args.mindnav_mode in ("kg", "heuristic"):
            kg_trace_logger.log_final(
                episode_idx=episode_number,
                goal_name=agent[0].goal_name,
                kg=kg,
                metrics=metrics,
                steps=agent[0].l_step,
            )
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        log += "Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
        print(log)
        logging.info(log)
        # ------------------------------------------------------------------


    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    return avg_metrics

if __name__ == "__main__":
    main()
