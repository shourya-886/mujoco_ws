import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

class ArduinobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("arduinobot_mjcf.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # Action space: 4 actuators, normalized ctrlrange
        self.action_space = spaces.Box(
            low=np.array([-1.5708, -1.5708, -1.5708, -1.5708]),
            high=np.array([1.5708, 1.5708, 1.5708, 0.0]),
            dtype=np.float32
        )

        # Observation: joint positions + velocities (5 joints, but joint_5 is coupled)
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0   # <-- define your task here
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()