import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer


MJCF_PATH = os.path.expanduser(
    "~/mujoco_ws/src/description/urdf/model.xml"
)

ARM_JOINT_NAME = "base_to_arm"
PENDULUM_JOINT_NAME = "arm_to_pendulum"
ARM_ACTUATOR_NAME = "base_to_arm_motor"


class FurutaPendulumEnv(gym.Env):
    """
    Gymnasium environment for the MuJoCo Furuta (rotary inverted) pendulum.

    Task: balance the pendulum upright.

    The physical MuJoCo pendulum angle is pi at the upright position.
    Internally, alpha is defined as the wrapped ANGULAR ERROR from upright,
    so:

        alpha = 0  -> perfectly upright (target)
        alpha = +/-pi -> hanging down

    Observation:
        [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]

    Action:
        Desired torque/command for the single arm actuator, in the
        actuator's own ctrlrange (matches the MJCF's <motor ctrlrange=.../>).

    Reward:
        R = -(theta^2
              + 0.01 * theta_dot^2
              + 0.001 * alpha^2
              + 0.00001 * alpha_dot^2)
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode=None,
        max_steps=1000,
        upright_threshold=0.17,
        fall_threshold=0.8,
        control_substeps=10,
        init_alpha_noise=0.05,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.viewer = None

        self.max_steps = max_steps
        self.upright_threshold = upright_threshold
        self.fall_threshold = fall_threshold
        self.control_substeps = control_substeps
        self.init_alpha_noise = init_alpha_noise

        self.steps = 0

        # ------------------------------------------------------------
        # Load MJCF as-is.
        # ------------------------------------------------------------
        if not os.path.exists(MJCF_PATH):
            raise FileNotFoundError(f"MJCF file not found:\n{MJCF_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------
        # Cache joint / actuator IDs.
        # ------------------------------------------------------------
        self.arm_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINT_NAME
        )
        self.pendulum_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, PENDULUM_JOINT_NAME
        )
        self.arm_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACTUATOR_NAME
        )

        if self.arm_joint_id == -1:
            raise RuntimeError(f"Joint '{ARM_JOINT_NAME}' not found in MJCF.")
        if self.pendulum_joint_id == -1:
            raise RuntimeError(f"Joint '{PENDULUM_JOINT_NAME}' not found in MJCF.")
        if self.arm_actuator_id == -1:
            raise RuntimeError(f"Actuator '{ARM_ACTUATOR_NAME}' not found in MJCF.")

        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_id]
        self.arm_qvel_adr = self.model.jnt_dofadr[self.arm_joint_id]
        self.pendulum_qpos_adr = self.model.jnt_qposadr[self.pendulum_joint_id]
        self.pendulum_qvel_adr = self.model.jnt_dofadr[self.pendulum_joint_id]

        self.arm_joint_range = self.model.jnt_range[self.arm_joint_id].copy()

        # ------------------------------------------------------------
        # Action space: single actuator, torque/command in its ctrlrange.
        # ------------------------------------------------------------
        ctrl_range = self.model.actuator_ctrlrange[self.arm_actuator_id]

        if self.model.actuator_ctrllimited[self.arm_actuator_id]:
            action_low = np.array([ctrl_range[0]], dtype=np.float32)
            action_high = np.array([ctrl_range[1]], dtype=np.float32)
        else:
            action_low = np.array([-1.0], dtype=np.float32)
            action_high = np.array([1.0], dtype=np.float32)

        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # ------------------------------------------------------------
        # Observation space:
        #   [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]
        #
        # alpha is the angular error from upright.
        # ------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    # ================================================================
    # OBSERVATION
    # ================================================================

    def _get_obs(self):

        theta = self.data.qpos[self.arm_qpos_adr]
        alpha = self._get_alpha()
        theta_dot = self.data.qvel[self.arm_qvel_adr]
        alpha_dot = self.data.qvel[self.pendulum_qvel_adr]

        observation = np.array(
            [
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                theta_dot,
                alpha_dot,
            ],
            dtype=np.float32,
        )

        return observation

    def _get_alpha(self):
        """
        Return pendulum angular error from the upright target.

        MuJoCo physical angle:
            pi  -> upright
            0   -> hanging down

        Internal alpha:
            0   -> upright target
            +/-pi -> hanging down
        """
        raw_alpha = self.data.qpos[self.pendulum_qpos_adr]

        # Shift the physical angle by pi so that upright becomes zero,
        # then wrap the result to [-pi, pi].
        return float(
            (raw_alpha - np.pi + np.pi) % (2 * np.pi) - np.pi
        )

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Arm starts centered.
        self.data.qpos[self.arm_qpos_adr] = 0.0
        self.data.qvel[self.arm_qvel_adr] = 0.0

        # Physical pendulum starts near upright (physical angle = pi).
        # The internal alpha error therefore starts near zero.
        self.data.qpos[self.pendulum_qpos_adr] = (
            np.pi + self.np_random.uniform(
                -self.init_alpha_noise, self.init_alpha_noise
            )
        )
        self.data.qvel[self.pendulum_qvel_adr] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0

        observation = self._get_obs()

        info = {
            "alpha": self._get_alpha(),
            "theta": float(self.data.qpos[self.arm_qpos_adr]),
        }

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        action = np.asarray(action, dtype=np.float32)

        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"but received {action.shape}."
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[self.arm_actuator_id] = action[0]

        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        alpha = self._get_alpha()
        theta = float(self.data.qpos[self.arm_qpos_adr])
        theta_dot = float(self.data.qvel[self.arm_qvel_adr])
        alpha_dot = float(self.data.qvel[self.pendulum_qvel_adr])

        # ------------------------------------------------------------
        # REWARD
        #
        # Exactly:
        #
        # R = -(theta^2
        #       + 0.01 * theta_dot^2
        #       + 0.001 * alpha^2
        #       + 0.00001 * alpha_dot^2)
        #
        # theta = arm rotation
        # alpha = pendulum angular error from upright
        #
        # Therefore alpha = 0 is the target upright position.
        # ------------------------------------------------------------
        reward = -(
            theta ** 2
            + 0.01 * theta_dot ** 2
            + 0.001 * alpha ** 2
            + 0.00001 * alpha_dot ** 2
        )

        # ------------------------------------------------------------
        # Termination.
        #
        # alpha = 0 is upright.
        # The episode ends when the pendulum is farther than
        # fall_threshold from upright or the arm reaches a limit.
        # ------------------------------------------------------------
        fell_over = abs(alpha) > self.fall_threshold

        hit_arm_limit = (
            theta <= self.arm_joint_range[0]
            or theta >= self.arm_joint_range[1]
        )

        terminated = bool(fell_over or hit_arm_limit)
        truncated = self.steps >= self.max_steps

        observation = self._get_obs()

        info = {
            "alpha": alpha,
            "theta": theta,
            "theta_dot": theta_dot,
            "alpha_dot": alpha_dot,
            "upright": abs(alpha) < self.upright_threshold,
            "fell_over": fell_over,
            "hit_arm_limit": hit_arm_limit,
            "steps": self.steps,
        }

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    # ================================================================
    # RENDER
    # ================================================================

    def render(self):

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    # ================================================================
    # CLOSE
    # ================================================================

    def close(self):

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ====================================================================
# TEST
# ====================================================================

if __name__ == "__main__":

    env = FurutaPendulumEnv(render_mode="human")

    observation, info = env.reset()

    print("=" * 60)
    print("MuJoCo Furuta Pendulum Environment Test")
    print("=" * 60)
    print(f"Action shape: {env.action_space.shape}")
    print(f"Observation shape: {env.observation_space.shape}")
    print(f"Initial alpha error: {info['alpha']:.4f} rad")
    print("Target alpha: 0.0000 rad (upright)")
    print("=" * 60)

    try:
        for episode in range(10):

            observation, info = env.reset()
            print(f"\nEpisode {episode + 1}")

            for step in range(env.max_steps):

                # Zero-action sanity check.
                action = np.zeros(env.action_space.shape, dtype=np.float32)

                observation, reward, terminated, truncated, info = env.step(action)

                if step % 50 == 0:
                    print(
                        f"Step {step:03d} | "
                        f"alpha error: {info['alpha']:.3f} rad | "
                        f"reward: {reward:.3f}"
                    )

                time.sleep(0.02)

                if terminated or truncated:
                    status = "FELL/LIMIT" if terminated else "TIMEOUT"
                    print(f"Episode ended: {status}")
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        env.close()
