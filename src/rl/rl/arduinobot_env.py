import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

# Absolute path to the MJCF so this works no matter where you run the script from.
MJCF_PATH = os.path.expanduser(
    "~/mujoco_ws/src/description/urdf/arduinobot_mjcf.xml"
)


class ArduinobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=200, success_threshold=0.03):
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.steps = 0

        # Action space: 4 actuators (servo_joint_1, servo_joint_2, servo_joint_3, servo_gripper)
        # Bounds taken from the actuator ctrlrange in the MJCF.
        self.action_space = spaces.Box(
            low=np.array([-1.5708, -1.5708, -1.5708, -1.5708], dtype=np.float32),
            high=np.array([1.5708, 1.5708, 1.5708, 0.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: joint positions + velocities + vector from gripper tip to target.
        # (5 joints -- joint_5 is mechanically coupled to joint_4 via the
        # equality constraint, but it still has its own qpos/qvel entry.)
        obs_dim = self.model.nq + self.model.nv + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        gripper_pos = self.data.site("gripper_tip").xpos
        target_pos = self.data.site("target").xpos
        to_target = target_pos - gripper_pos
        return np.concatenate([self.data.qpos, self.data.qvel, to_target]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        gripper_pos = self.data.site("gripper_tip").xpos
        target_pos = self.data.site("target").xpos
        distance = float(np.linalg.norm(gripper_pos - target_pos))

        reward = -distance
        terminated = distance < self.success_threshold
        if terminated:
            reward += 10.0  # bonus for reaching the target
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"distance": distance}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    # Quick manual smoke test: random actions, rendered live.
    # Watch the printed distance -- it should vary as the arm moves, and you
    # should see the red gripper_tip site and green target site in the viewer.
    env = ArduinobotEnv(render_mode="human")
    obs, info = env.reset()

    for i in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 50 == 0:
            print(f"step {i}: distance={info['distance']:.3f}  reward={reward:.3f}")
        time.sleep(0.02)
        if terminated or truncated:
            print("episode ended, resetting" + (" (SUCCESS)" if terminated else " (timeout)"))
            obs, info = env.reset()

    env.close()