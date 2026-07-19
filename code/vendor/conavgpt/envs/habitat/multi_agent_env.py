import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat
from pathlib import Path

from constants import category_to_id, hm3d_category
import utils.pose as pu

coco_categories = [0, 3, 2, 4, 5, 1]

class Multi_Agent_Env(habitat.Env):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, config_env, use_gtsem=False):

        super().__init__(config_env)
        self.use_gtsem = bool(use_gtsem)

        # Initializations
        self.episode_no = 0
   
        fileName = (
            Path(__file__).resolve().parents[4]
            / 'configs' / 'matterport_category_mappings.tsv'
        )

        text = ''
        lines = []
        items = []
        self.hm3d_semantic_mapping={}

        with fileName.open('r') as f:
            text = f.read()
        lines = text.split('\n')

        for l in lines:
            items.append(l.split('    '))

        for i in items:
            if len(i) > 3:
                self.hm3d_semantic_mapping[i[2]] = i[-1]

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """


        self.episode_no += 1

        obs = super().reset()
        if self.use_gtsem:
            self.scene = self.sim.semantic_annotations()
            if not getattr(self.scene, "objects", None):
                raise RuntimeError(
                    "GT semantic mode found no Habitat semantic annotations; "
                    "use the mindnav38_hm3d022 environment"
                )
            for agent_obs in obs:
                agent_obs["semantic"] = self._preprocess_semantic(
                    agent_obs["semantic"]
                )
  
        # rgb = obs['rgb'].astype(np.uint8)
        # depth = obs['depth']
        # semantic = self._preprocess_semantic(obs["semantic"])

        # state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
   
        return obs

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        obs = super().step(action)
        if self.use_gtsem:
            for agent_obs in obs:
                agent_obs["semantic"] = self._preprocess_semantic(
                    agent_obs["semantic"]
                )

        # rgb = obs['rgb'].astype(np.uint8)
        # depth = obs['depth']
        # semantic = self._preprocess_semantic(obs["semantic"])
        # state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)

        return obs

    def _preprocess_semantic(self, semantic):
        """Map Habitat instance IDs to MindNav's 15 semantic channels.

        The conversion reads from an immutable copy so low-valued class IDs
        cannot be mistaken for yet-unprocessed Habitat instance IDs.
        Unknown categories use channel 15, which mapping treats as background.
        """
        instance_map = np.asarray(semantic).squeeze()
        category_map = np.full(instance_map.shape, 15, dtype=np.uint8)
        objects = getattr(self.scene, "objects", [])
        category_index = {name: idx for idx, name in enumerate(hm3d_category)}
        for instance_id in np.unique(instance_map):
            instance_id = int(instance_id)
            if instance_id < 0 or instance_id >= len(objects):
                continue
            obj = objects[instance_id]
            if obj is None or getattr(obj, "category", None) is None:
                continue
            raw_name = obj.category.name()
            mapped_name = self.hm3d_semantic_mapping.get(raw_name, raw_name)
            channel = category_index.get(mapped_name)
            if channel is not None:
                category_map[instance_map == instance_id] = channel
        return category_map

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
