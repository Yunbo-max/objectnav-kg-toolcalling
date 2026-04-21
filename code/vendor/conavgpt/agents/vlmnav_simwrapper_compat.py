"""
Compatible SimWrapper for habitat 0.2.1.
Same interface as VLMNav's SimWrapper but uses our working sim setup.
"""
import habitat_sim
import numpy as np
from agents.vlmnav_simwrapper import PolarAction

try:
    import magnum as mn
except ImportError:
    mn = None

from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis


class SimWrapperCompat:
    """Drop-in replacement for VLMNav's SimWrapper that works with habitat 0.2.1."""

    def __init__(self, scene_path, gpu_id=0, sensor_cfg=None):
        if sensor_cfg is None:
            sensor_cfg = {'height': 1.5, 'pitch': -0.45, 'res_factor': 2, 'fov': 131}

        self.resolution = (1080 // sensor_cfg['res_factor'], 1920 // sensor_cfg['res_factor'])

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = gpu_id
        sim_cfg.scene_id = scene_path

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.radius = 0.17
        agent_cfg.height = 1.5

        # RGB sensor
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = list(self.resolution)
        rgb_spec.hfov = sensor_cfg['fov']
        rgb_spec.position = [0, sensor_cfg['height'], 0]
        rgb_spec.orientation = [sensor_cfg['pitch'], 0, 0]

        # Depth sensor
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = list(self.resolution)
        depth_spec.hfov = sensor_cfg['fov']
        depth_spec.position = [0, sensor_cfg['height'], 0]
        depth_spec.orientation = [sensor_cfg['pitch'], 0, 0]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)

    def set_state(self, pos, rotation):
        """Set agent position and rotation."""
        agent = self.sim.get_agent(0)
        state = habitat_sim.AgentState()
        state.position = np.array(pos)
        if hasattr(rotation, 'w'):
            state.rotation = rotation
        else:
            state.rotation = rotation
        agent.set_state(state)

    def get_obs(self):
        """Get current observation."""
        obs = self.sim.get_sensor_observations(0)
        obs['agent_state'] = self.sim.get_agent(0).get_state()
        return obs

    def step(self, action: PolarAction):
        """Execute polar action (r, theta) using pathfinder."""
        agent = self.sim.get_agent(0)
        curr_state = agent.get_state()

        if action is PolarAction.null or action is PolarAction.stop:
            obs = self.sim.get_sensor_observations(0)
            obs['agent_state'] = curr_state
            return obs

        new_state = habitat_sim.AgentState()
        new_state.position = np.copy(curr_state.position)
        new_state.rotation = curr_state.rotation

        # Rotate
        theta, axis = quat_to_angle_axis(curr_state.rotation)
        if axis[1] < 0:
            theta = 2 * np.pi - theta
        new_theta = theta + action.theta
        new_state.rotation = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))

        # Move forward
        local_point = np.array([0, 0, -action.r])
        # Convert local to global
        sin_t = np.sin(new_theta)
        cos_t = np.cos(new_theta)
        global_offset = np.array([
            local_point[0] * cos_t + local_point[2] * sin_t,
            local_point[1],
            -local_point[0] * sin_t + local_point[2] * cos_t
        ])
        target_pos = curr_state.position + global_offset

        # Use pathfinder for collision-aware movement
        delta = (target_pos - curr_state.position) / 10
        new_pos = np.copy(curr_state.position)
        for _ in range(10):
            new_pos = self.sim.pathfinder.try_step(new_pos, new_pos + delta)

        new_state.position = new_pos
        agent.set_state(new_state)

        obs = self.sim.get_sensor_observations(0)
        obs['agent_state'] = agent.get_state()
        return obs

    def get_path(self, path_calc):
        """Get geodesic distance."""
        if self.sim.pathfinder.find_path(path_calc):
            return path_calc.geodesic_distance
        return 1000

    def reset(self):
        try:
            self.sim.close()
        except:
            pass
