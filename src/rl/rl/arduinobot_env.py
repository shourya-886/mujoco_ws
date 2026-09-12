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

    Task: balance the pendulum upright (alpha = 0) by driving the arm motor.

    Observation:
        [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]

    Action:
        Desired torque/command for the single arm actuator, in the
        actuator's own ctrlrange (matches the MJCF's <motor ctrlrange=.../>).
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode=None,
        max_steps=1000,
        upright_threshold=0.17,   # ~10 deg, alpha within this counts as "up"
        fall_threshold=0.8,       # ~46 deg, alpha beyond this ends the episode
        control_substeps=10,      # sim steps per env.step() call
        init_alpha_noise=0.05,    # rad, randomize start near hanging-down
        velocity_penalty_weight=0.01,
        action_penalty_weight=0.001,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.viewer = None

        self.max_steps = max_steps
        self.upright_threshold = upright_threshold
        self.fall_threshold = fall_threshold
        self.control_substeps = control_substeps
        self.init_alpha_noise = init_alpha_noise
        self.velocity_penalty_weight = velocity_penalty_weight
        self.action_penalty_weight = action_penalty_weight

        self.steps = 0

        # ------------------------------------------------------------
        # Load MJCF as-is. No injected bodies needed for this task.
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
        # ------------------------------------------------------------
        obs_dim = 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ================================================================
    # OBSERVATION
    # ================================================================

    def _get_obs(self):

        theta = self.data.qpos[self.arm_qpos_adr]
        alpha = self.data.qpos[self.pendulum_qpos_adr]
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
        # Wrapped to [-pi, pi]. NOTE: alpha == 0 here means hanging DOWN
        # (the stable equilibrium), not upright. Use
        # _get_upright_distance() below for "how far from balanced".
        raw_alpha = self.data.qpos[self.pendulum_qpos_adr]
        return float(
            (raw_alpha + np.pi) % (2 * np.pi) - np.pi
        )

    def _get_upright_distance(self, alpha):
        # Angular distance from upright (+/- pi), wrapped to [0, pi].
        # 0 = perfectly upright, pi = hanging straight down.
        return float(np.pi - abs(alpha))

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Arm starts centered.
        self.data.qpos[self.arm_qpos_adr] = 0.0
        self.data.qvel[self.arm_qvel_adr] = 0.0

        # Pendulum starts near upright (alpha = +/- pi is upright, alpha = 0
        # is the hanging-down stable equilibrium — confirmed by testing:
        # zero action holds the pendulum steady at alpha = 0, which is
        # only possible if that's the stable/hanging point, not upright).
        # Small random perturbation so the policy doesn't overfit to one
        # exact starting state.
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
        # Reward
        #
        # 1. Upright reward: peaks at alpha = 0, standard cos-based shaping.
        # 2. Velocity penalty: discourages wild spinning at the top.
        # 3. Action penalty: discourages unnecessary torque.
        # 4. Arm-limit penalty: discourages hugging the mechanical stops.
        # ------------------------------------------------------------
        # Confirmed by testing: alpha = 0 is the pendulum's stable
        # hanging-down equilibrium (zero action holds steady there).
        # Upright (unstable equilibrium) is therefore alpha = +/- pi.
        # -cos(alpha) peaks at alpha = +/-pi (upright) and is at its
        # minimum at alpha = 0 (hanging), which is what we want to reward.
        upright_reward = -np.cos(alpha)  # +1 at top (pi), -1 at bottom (0)

        velocity_penalty = self.velocity_penalty_weight * (
            theta_dot ** 2 + alpha_dot ** 2
        )

        normalized_action = action[0] / (
            self.action_space.high[0] + 1e-8
        )
        action_penalty = self.action_penalty_weight * (normalized_action ** 2)

        arm_limit_margin = min(
            theta - self.arm_joint_range[0],
            self.arm_joint_range[1] - theta,
        )
        arm_limit_penalty = 0.0
        if arm_limit_margin < 0.2:  # ~11 deg from a hard stop
            arm_limit_penalty = 2.0 * (0.2 - arm_limit_margin)

        reward = (
            upright_reward
            - velocity_penalty
            - action_penalty
            - arm_limit_penalty
        )

        # ------------------------------------------------------------
        # Termination.
        #
        # "Fallen" means too far from upright (+/- pi), NOT too far from
        # zero — alpha = 0 is the hanging-down equilibrium, not a fall.
        # ------------------------------------------------------------
        upright_distance = self._get_upright_distance(alpha)
        fell_over = upright_distance > self.fall_threshold
        hit_arm_limit = (
            theta <= self.arm_joint_range[0]
            or theta >= self.arm_joint_range[1]
        )

        terminated = bool(fell_over or hit_arm_limit)

        if terminated:
            reward -= 10.0

        truncated = self.steps >= self.max_steps

        observation = self._get_obs()

        info = {
            "alpha": alpha,
            "theta": theta,
            "theta_dot": theta_dot,
            "alpha_dot": alpha_dot,
            "upright": upright_distance < self.upright_threshold,
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
    print(f"Initial alpha: {info['alpha']:.4f} rad")
    print("=" * 60)

    try:
        for episode in range(10):

            observation, info = env.reset()
            print(f"\nEpisode {episode + 1}")

            for step in range(env.max_steps):

                # Zero action sanity check: with no torque, the pendulum
                # starts at rest near upright and should stay there for a
                # while (small numerical drift aside). If it falls
                # immediately even with zero action, something is wrong
                # with the env itself. Switch to env.action_space.sample()
                # to see how fast a random policy fails (expected: fast).
                action = np.zeros(env.action_space.shape, dtype=np.float32)

                observation, reward, terminated, truncated, info = env.step(action)

                if step % 50 == 0:
                    print(
                        f"Step {step:03d} | "
                        f"alpha: {info['alpha']:.3f} rad | "
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