import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat

from constants import DATASET_SEMANTIC_CATEGORIES
import utils.pose as pu

MP3D_CATEGORY_INDEX = {
    name: idx for idx, name in enumerate(DATASET_SEMANTIC_CATEGORIES["mp3d"])
}
HM3D_CATEGORY_INDEX = {
    name: idx for idx, name in enumerate(DATASET_SEMANTIC_CATEGORIES["hm3d"])
}
MP3D_CATEGORY_ALIASES = {
    "couch": "sofa",
    "tv": "tv_monitor",
    "television": "tv_monitor",
    "gym equipment": "gym_equipment",
    "chest of drawers": "chest_of_drawers",
    "chest-of-drawers": "chest_of_drawers",
    "dining table": "table",
    "dining-table": "table",
    "potted plant": "plant",
}


def _canonical_category_name(name):
    name = str(name or "").strip().lower()
    name = MP3D_CATEGORY_ALIASES.get(name, name)
    return name.replace(" ", "_").replace("-", "_")

class Multi_Agent_Env(habitat.Env):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, config_env):

        super().__init__(config_env)

        # Initializations
        self.episode_no = 0
        dataset_cfg = getattr(config_env, "DATASET", None)
        data_path = str(getattr(dataset_cfg, "DATA_PATH", "") or "").lower()
        episodes_dir = str(getattr(dataset_cfg, "EPISODES_DIR", "") or "").lower()
        self.is_mp3d_task = "mp3d" in data_path or "mp3d" in episodes_dir
        self.semantic_lookup = None
   
        fileName = 'data/matterport_category_mappings.tsv'

        text = ''
        lines = []
        items = []
        self.hm3d_semantic_mapping={}

        with open(fileName, 'r') as f:
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
        self.scene = self.sim.semantic_annotations()
        self.semantic_lookup = self._build_semantic_lookup()
  
        for i in range(len(obs)):
            obs[i]['semantic'] = self._preprocess_semantic(obs[i]["semantic"])

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


        for i in range(len(obs)):
            obs[i]['semantic'] = self._preprocess_semantic(obs[i]["semantic"])

        return obs

    def _category_id_for_name(self, raw_name):
        mapped_name = self.hm3d_semantic_mapping.get(raw_name, raw_name)
        category_name = _canonical_category_name(mapped_name)
        category_index = MP3D_CATEGORY_INDEX if self.is_mp3d_task else HM3D_CATEGORY_INDEX
        if category_name in category_index:
            return category_index[category_name]
        return 255

    def _build_semantic_lookup(self):
        lookup = np.full(len(self.scene.objects), 255, dtype=np.uint8)
        for raw_id, obj in enumerate(self.scene.objects):
            if obj is None or obj.category is None:
                continue
            lookup[raw_id] = self._category_id_for_name(obj.category.name())
        return lookup

    def _preprocess_semantic(self, semantic):
        raw_semantic = semantic.astype(np.int64, copy=False)
        semantic_out = np.full(raw_semantic.shape, 255, dtype=np.uint8)
        lookup = self.semantic_lookup
        if lookup is None:
            lookup = self._build_semantic_lookup()
            self.semantic_lookup = lookup
        valid = np.logical_and(raw_semantic >= 0, raw_semantic < len(lookup))
        semantic_out[valid] = lookup[raw_semantic[valid]]
        return np.expand_dims(semantic_out, 2)

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
