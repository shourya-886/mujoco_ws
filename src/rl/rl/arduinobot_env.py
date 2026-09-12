import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

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

        self.action_space = spaces.Box(
            low=np.array([-1.5708, -1.5708, -1.5708, -1.5708], dtype=np.float32),
            high=np.array([1.5708, 1.5708, 1.5708, 0.0], dtype=np.float32),
            dtype=np.float32,
        )

        # qpos + qvel + vector-to-target(3) + distance(1) + explicit encoder block(4)
        obs_dim = self.model.nq + self.model.nv + 3 + 1 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        gripper_pos = self.data.site("gripper_tip").xpos
        target_pos = self.data.site("target").xpos
        to_target = target_pos - gripper_pos
        distance = np.array([np.linalg.norm(to_target)], dtype=np.float32)

        # Explicit "encoder" readout: current angle of each actuated joint
        # (joint_1..joint_4). Redundant with qpos but kept as its own clearly
        # labeled block -- useful if this ever needs to map directly onto
        # real servo positions for sim-to-real work later.
        joint_positions = self.data.qpos[:4].copy()

        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
            to_target,
            distance,
            joint_positions,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Guarantee the target is reachable AND above the floor: sample
        # random joint angles within their real limits, run forward
        # kinematics, and reject/resample any candidate that lands below the
        # floor plane (unreachable in practice -- the floor geom blocks it).
        j1_range = self.model.jnt_range[0]
        j2_range = self.model.jnt_range[1]
        j3_range = self.model.jnt_range[2]

        min_z = 0.05  # small margin above the floor plane
        max_attempts = 50
        reachable_point = None
        candidate = None

        for _ in range(max_attempts):
            sample_qpos = self.data.qpos.copy()
            sample_qpos[0] = self.np_random.uniform(j1_range[0], j1_range[1])
            sample_qpos[1] = self.np_random.uniform(j2_range[0], j2_range[1])
            sample_qpos[2] = self.np_random.uniform(j3_range[0], j3_range[1])

            self.data.qpos[:] = sample_qpos
            mujoco.mj_forward(self.model, self.data)
            candidate = self.data.site("gripper_tip").xpos.copy()

            if candidate[2] >= min_z:
                reachable_point = candidate
                break

        if reachable_point is None:
            # Extremely unlikely fallback after max_attempts: clamp z upward.
            reachable_point = candidate.copy()
            reachable_point[2] = max(reachable_point[2], min_z)

        mujoco.mj_resetData(self.model, self.data)
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.model.site_pos[target_id] = reachable_point
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_gripper_pos = self.data.site("gripper_tip").xpos.copy()

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        gripper_pos = self.data.site("gripper_tip").xpos
        target_pos = self.data.site("target").xpos
        distance = float(np.linalg.norm(gripper_pos - target_pos))
        prev_distance = float(np.linalg.norm(prev_gripper_pos - target_pos))

        # Dense shaping: reward reducing distance each step (progress), not
        # just being close overall -- this is the main fix for the arm
        # struggling to converge with a purely sparse/near-sparse signal.
        progress = prev_distance - distance
        reward = -distance * 0.1 + progress * 10.0

        # Small control penalty discourages jittery/wasteful motion.
        reward -= 0.001 * float(np.sum(np.square(action)))

        terminated = distance < self.success_threshold
        if terminated:
            reward += 20.0
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