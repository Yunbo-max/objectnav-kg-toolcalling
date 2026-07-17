#!/usr/bin/env python3
"""Audit plant/TV perception at HM3D ObjectNav oracle viewpoints.

This is a passive diagnostic: it teleports agent 0 to dataset-provided goal
viewpoints and runs the same RedNet and Mask R-CNN models used by MindNav. It
does not invoke the Brain, FMM, or navigation actions.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
VENDOR_ROOT = os.path.join(REPO_ROOT, "code", "vendor", "conavgpt")
MULTI_ROBOT_ROOT = os.path.join(VENDOR_ROOT, "multi-robot-setting")
for path in (VENDOR_ROOT, MULTI_ROBOT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2  # noqa: E402
import habitat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from constants import mp_categories_mapping  # noqa: E402
from envs.habitat.multi_agent_env import Multi_Agent_Env  # noqa: E402
from RedNet.RedNet_model import load_rednet  # noqa: E402
from utils.semantic_prediction import SemanticPredMaskRCNN  # noqa: E402
from habitat.sims.habitat_simulator.actions import HabitatSimActions  # noqa: E402


TARGET_TO_SEMANTIC_INDEX = {
    "plant": 2,
    "tv_monitor": 5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_mini")
    parser.add_argument("--sim_gpu_id", type=int, default=0)
    parser.add_argument("--sem_gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--pitch_steps",
        default="0",
        help="comma-separated 30-degree sensor pitch steps; positive looks up",
    )
    parser.add_argument("--sem_pred_prob_thr", type=float, default=0.9)
    parser.add_argument(
        "--targets", default="plant,tv_monitor",
        help="comma-separated ObjectNav targets",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def mask_stats(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    pixels = int(binary.sum())
    if pixels == 0:
        return {
            "pixels": 0,
            "components": 0,
            "largest_component": 0,
            "largest_bbox_xywh": None,
        }
    labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    foreground = stats[1:]
    largest = foreground[int(np.argmax(foreground[:, cv2.CC_STAT_AREA]))]
    return {
        "pixels": pixels,
        "components": int(labels - 1),
        "largest_component": int(largest[cv2.CC_STAT_AREA]),
        "largest_bbox_xywh": [
            int(largest[cv2.CC_STAT_LEFT]),
            int(largest[cv2.CC_STAT_TOP]),
            int(largest[cv2.CC_STAT_WIDTH]),
            int(largest[cv2.CC_STAT_HEIGHT]),
        ],
    }


def make_overlay(rgb, mask, label, color):
    output = rgb.copy()
    selected = mask.astype(bool)
    output[selected] = (
        0.45 * output[selected] + 0.55 * np.asarray(color)
    ).astype(np.uint8)
    cv2.putText(
        output,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def save_view_artifact(output_dir, record, rgb, rednet, maskrcnn):
    scene_dir = os.path.join(
        output_dir,
        record["scene"].replace(".basis.glb", ""),
        record["target"],
    )
    os.makedirs(scene_dir, exist_ok=True)
    stem = (
        f"object_{record['object_id']}_rank_{record['view_rank']:02d}_"
        f"pitch_{record['pitch_degrees']:+03d}"
    )
    fused = np.logical_or(rednet, maskrcnn).astype(np.uint8)
    panels = [
        rgb,
        make_overlay(rgb, rednet, "RedNet", (255, 0, 0)),
        make_overlay(rgb, maskrcnn, "Mask R-CNN", (0, 0, 255)),
        make_overlay(rgb, fused, "Fused", (255, 0, 255)),
    ]
    image_path = os.path.join(scene_dir, stem + ".png")
    cv2.imwrite(image_path, np.concatenate(panels, axis=1)[:, :, ::-1])
    record["image"] = image_path


def get_pitched_observation(env, view_point, pitch_steps):
    """Render one goal viewpoint after an independent sensor pitch action."""
    env.sim.set_agent_state(
        view_point.agent_state.position,
        view_point.agent_state.rotation,
        agent_id=0,
        reset_sensors=True,
    )
    if pitch_steps == 0:
        return env.sim.get_observations_at()

    action = (
        HabitatSimActions.LOOK_UP
        if pitch_steps > 0
        else HabitatSimActions.LOOK_DOWN
    )
    observations = None
    for _ in range(abs(pitch_steps)):
        observations = env.sim.step(
            [int(action)] * len(env.sim.habitat_config.AGENTS)
        )
    return observations[0] if observations is not None else None


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    pitch_steps = [
        int(value.strip())
        for value in args.pitch_steps.split(",")
        if value.strip()
    ]
    if not pitch_steps or any(abs(value) > 2 for value in pitch_steps):
        raise ValueError("--pitch_steps must contain values in [-2, 2]")
    targets = {
        target.strip() for target in args.targets.split(",") if target.strip()
    }
    unsupported = targets - set(TARGET_TO_SEMANTIC_INDEX)
    if unsupported:
        raise ValueError(f"Unsupported targets: {sorted(unsupported)}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "oracle_view_audit.jsonl")
    open(jsonl_path, "w").close()

    original_cwd = os.getcwd()
    os.chdir(VENDOR_ROOT)
    env = None
    try:
        config = habitat.get_config(
            config_paths=["envs/habitat/configs/tasks/multi_objectnav_hm3d.yaml"]
        )
        config.defrost()
        config.DATASET.SPLIT = args.split
        config.DATASET.DATA_PATH = config.DATASET.DATA_PATH.replace(
            "{split}", args.split
        )
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id
        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.freeze()
        env = Multi_Agent_Env(config_env=config)
        env.seed(args.seed)

        device = torch.device(f"cuda:{args.sem_gpu_id}")
        semantic_args = SimpleNamespace(
            sem_pred_prob_thr=args.sem_pred_prob_thr,
            sem_gpu_id=args.sem_gpu_id,
            visualize=0,
        )
        maskrcnn_model = SemanticPredMaskRCNN(semantic_args)
        rednet_model = load_rednet(
            device,
            ckpt="RedNet/model/rednet_semmap_mp3d_40.pth",
            resize=True,
        )
        rednet_model.eval().to(device)

        processed_scene_targets = set()
        for _ in range(env.number_of_episodes):
            env.reset()
            episode = env.current_episode
            target = episode.object_category
            scene = os.path.basename(episode.scene_id)
            scene_target = (scene, target)
            if target not in targets or scene_target in processed_scene_targets:
                continue
            processed_scene_targets.add(scene_target)
            semantic_index = TARGET_TO_SEMANTIC_INDEX[target]

            for goal_index, goal in enumerate(episode.goals):
                view_points = sorted(
                    goal.view_points,
                    key=lambda view: float(getattr(view, "iou", 0.0)),
                    reverse=True,
                )[:args.top_k]
                for rank, view_point in enumerate(view_points, start=1):
                    for pitch_step in pitch_steps:
                        observation = get_pitched_observation(
                            env, view_point, pitch_step
                        )
                        if observation is None:
                            continue
                        rgb = observation["rgb"].astype(np.uint8)
                        depth = observation["depth"]
                        with torch.no_grad():
                            red_prediction = rednet_model(
                                torch.from_numpy(rgb)
                                .to(device)
                                .unsqueeze(0)
                                .float(),
                                torch.from_numpy(depth)
                                .to(device)
                                .unsqueeze(0)
                                .float(),
                            ).squeeze().cpu().numpy()
                        mask_prediction, _ = maskrcnn_model.get_prediction(rgb)
                        red_target = (
                            red_prediction
                            == mp_categories_mapping[semantic_index]
                        )
                        mask_target = (
                            mask_prediction[:, :, semantic_index] > 0
                        )
                        fused_target = np.logical_or(
                            red_target, mask_target
                        )

                        record = {
                            "schema_version": 1,
                            "split": args.split,
                            "scene": scene,
                            "episode_id_used_for_scene": str(
                                episode.episode_id
                            ),
                            "target": target,
                            "goal_index": int(goal_index),
                            "object_id": int(goal.object_id),
                            "view_rank": int(rank),
                            "view_iou": float(
                                getattr(view_point, "iou", 0.0)
                            ),
                            "pitch_steps": int(pitch_step),
                            "pitch_degrees": int(pitch_step * 30),
                            "rednet": mask_stats(red_target),
                            "maskrcnn": mask_stats(mask_target),
                            "fused_union": mask_stats(fused_target),
                            "source_intersection_pixels": int(
                                np.logical_and(
                                    red_target, mask_target
                                ).sum()
                            ),
                        }
                        save_view_artifact(
                            output_dir,
                            record,
                            rgb,
                            red_target,
                            mask_target,
                        )
                        with open(jsonl_path, "a") as output_file:
                            json.dump(
                                record, output_file, ensure_ascii=False
                            )
                            output_file.write("\n")
                        print(json.dumps(record, ensure_ascii=False))

        print(f"Wrote oracle-view audit to {jsonl_path}")
    finally:
        if env is not None:
            env.close()
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
