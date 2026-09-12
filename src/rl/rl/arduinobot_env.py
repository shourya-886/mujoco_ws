import os
import time
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer


MJCF_PATH = os.path.expanduser(
    "~/mujoco_ws/src/description/urdf/so101_mjcf.xml"
)

END_EFFECTOR_SITE = "gripperframe"
TARGET_BODY_NAME = "rl_target"
TARGET_SITE_NAME = "target"


class ArduinobotEnv(gym.Env):
    """
    Gymnasium environment for the MuJoCo ArduinoBot / SO-101 arm.

    Robot actuators are detected automatically from the MJCF, so the
    environment follows the joint configuration defined by the robot model.

    Observation:
        [joint_positions,
         joint_velocities,
         vector_to_target (x, y, z),
         distance_to_target,
         explicit_joint_encoder_positions]

    Action:
        Desired position for every actuator in the MuJoCo model.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode=None,
        max_steps=200,
        success_threshold=0.05,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.viewer = None

        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.steps = 0

        # ------------------------------------------------------------
        # Load MJCF with an RL target added automatically.
        # ------------------------------------------------------------
        self._temporary_mjcf_path = self._create_rl_mjcf()

        self.model = mujoco.MjModel.from_xml_path(
            self._temporary_mjcf_path
        )
        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------
        # Cache IDs
        # ------------------------------------------------------------
        self.end_effector_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            END_EFFECTOR_SITE,
        )

        self.target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            TARGET_BODY_NAME,
        )

        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            TARGET_SITE_NAME,
        )

        if self.end_effector_site_id == -1:
            raise RuntimeError(
                f"End-effector site '{END_EFFECTOR_SITE}' "
                f"was not found in the MJCF."
            )

        if self.target_body_id == -1:
            raise RuntimeError(
                f"Target body '{TARGET_BODY_NAME}' "
                f"was not created correctly."
            )

        if self.target_site_id == -1:
            raise RuntimeError(
                f"Target site '{TARGET_SITE_NAME}' "
                f"was not created correctly."
            )

        # ------------------------------------------------------------
        # Detect actuators automatically.
        # ------------------------------------------------------------
        self.num_actuators = self.model.nu

        if self.num_actuators == 0:
            raise RuntimeError(
                "The MuJoCo model contains no actuators."
            )

        self.actuator_names = []

        action_low = []
        action_high = []

        for actuator_id in range(self.num_actuators):
            actuator_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )

            self.actuator_names.append(actuator_name)

            ctrl_range = self.model.actuator_ctrlrange[actuator_id]

            # Use actuator control limits when available.
            if self.model.actuator_ctrllimited[actuator_id]:
                low = ctrl_range[0]
                high = ctrl_range[1]
            else:
                # Fallback for an unlimited actuator.
                low = -np.pi
                high = np.pi

            action_low.append(low)
            action_high.append(high)

        self.action_space = spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # Observation space
        #
        # qpos
        # qvel
        # vector to target: 3
        # distance: 1
        # explicit joint/encoder positions
        # ------------------------------------------------------------
        self.num_joint_positions = self.model.nq
        self.num_joint_velocities = self.model.nv

        obs_dim = (
            self.num_joint_positions
            + self.num_joint_velocities
            + 3
            + 1
            + self.num_actuators
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # Cache actuator -> joint qpos addresses.
        # ------------------------------------------------------------
        self.actuator_qpos_indices = []

        for actuator_id in range(self.num_actuators):

            joint_id = self.model.actuator_trnid[
                actuator_id, 0
            ]

            if joint_id < 0:
                raise RuntimeError(
                    f"Actuator '{self.actuator_names[actuator_id]}' "
                    "is not connected to a joint."
                )

            qpos_address = self.model.jnt_qposadr[joint_id]

            self.actuator_qpos_indices.append(
                qpos_address
            )

        self.actuator_qpos_indices = np.array(
            self.actuator_qpos_indices,
            dtype=np.int32,
        )

        # ------------------------------------------------------------
        # Cache controllable joint ranges.
        # ------------------------------------------------------------
        self.joint_ranges = []

        for actuator_id in range(self.num_actuators):

            joint_id = self.model.actuator_trnid[
                actuator_id, 0
            ]

            joint_range = self.model.jnt_range[
                joint_id
            ].copy()

            self.joint_ranges.append(joint_range)

        self.joint_ranges = np.array(
            self.joint_ranges,
            dtype=np.float64,
        )

        # Initial neutral configuration.
        self.initial_qpos = np.zeros(
            self.model.nq,
            dtype=np.float64,
        )

    # ================================================================
    # MJCF PREPARATION
    # ================================================================

    def _create_rl_mjcf(self):
        """
        Load the original MJCF and add a mocap target body/site.

        The uploaded robot MJCF does not contain a target object, so the
        RL environment adds one automatically without requiring manual
        modification of the robot model.
        """

        if not os.path.exists(MJCF_PATH):
            raise FileNotFoundError(
                f"MJCF file not found:\n{MJCF_PATH}"
            )

        tree = ET.parse(MJCF_PATH)
        root = tree.getroot()

        worldbody = root.find("worldbody")

        if worldbody is None:
            raise RuntimeError(
                "The MJCF does not contain a <worldbody> element."
            )

        # Remove an old RL target if one already exists.
        for body in worldbody.findall("body"):
            if body.get("name") == TARGET_BODY_NAME:
                worldbody.remove(body)

        # ------------------------------------------------------------
        # Add mocap target.
        #
        # A mocap body can be repositioned directly through data.mocap_pos.
        # ------------------------------------------------------------
        target_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": TARGET_BODY_NAME,
                "mocap": "true",
                "pos": "0.20 0.00 0.20",
            },
        )

        ET.SubElement(
            target_body,
            "geom",
            {
                "name": "target_visual",
                "type": "sphere",
                "size": "0.025",
                "rgba": "1 0 0 0.7",
                "contype": "0",
                "conaffinity": "0",
            },
        )

        ET.SubElement(
            target_body,
            "site",
            {
                "name": TARGET_SITE_NAME,
                "type": "sphere",
                "size": "0.012",
                "rgba": "0 1 0 1",
            },
        )

        # ------------------------------------------------------------
        # Save a temporary MJCF beside the original file.
        # This preserves relative asset/mesh paths.
        # ------------------------------------------------------------
        mjcf_directory = os.path.dirname(MJCF_PATH)

        temporary_file = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix="_rl.xml",
            prefix="arduinobot_",
            dir=mjcf_directory,
            delete=False,
        )

        temporary_path = temporary_file.name
        temporary_file.close()

        tree.write(
            temporary_path,
            encoding="utf-8",
            xml_declaration=True,
        )

        return temporary_path

    # ================================================================
    # OBSERVATION
    # ================================================================

    def _get_end_effector_position(self):
        return self.data.site_xpos[
            self.end_effector_site_id
        ].copy()

    def _get_target_position(self):
        return self.data.site_xpos[
            self.target_site_id
        ].copy()

    def _get_obs(self):

        end_effector_pos = (
            self._get_end_effector_position()
        )

        target_pos = (
            self._get_target_position()
        )

        to_target = (
            target_pos - end_effector_pos
        )

        distance = np.array(
            [
                np.linalg.norm(to_target)
            ],
            dtype=np.float32,
        )

        # Explicit encoder-style readout.
        #
        # These are the positions of the joints controlled by
        # the MuJoCo actuators.
        joint_positions = self.data.qpos[
            self.actuator_qpos_indices
        ].copy()

        observation = np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                to_target,
                distance,
                joint_positions,
            ]
        )

        return observation.astype(np.float32)

    # ================================================================
    # TARGET SAMPLING
    # ================================================================

    def _sample_reachable_target(self):
        """
        Generate a reachable target by:

        1. Sampling valid joint angles.
        2. Running forward kinematics.
        3. Taking the resulting end-effector position.

        This guarantees that the sampled target is inside the robot's
        reachable workspace.
        """

        min_height = 0.03
        max_attempts = 100

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        reachable_position = None

        for _ in range(max_attempts):

            sample_qpos = (
                self.initial_qpos.copy()
            )

            # Sample each actuated joint inside its real joint limits.
            for actuator_index, qpos_index in enumerate(
                self.actuator_qpos_indices
            ):

                joint_range = self.joint_ranges[
                    actuator_index
                ]

                low = joint_range[0]
                high = joint_range[1]

                # If the joint is effectively unlimited, use actuator range.
                if not np.isfinite(low) or not np.isfinite(high):
                    low = self.action_space.low[
                        actuator_index
                    ]
                    high = self.action_space.high[
                        actuator_index
                    ]

                sample_qpos[qpos_index] = (
                    self.np_random.uniform(low, high)
                )

            # --------------------------------------------------------
            # Keep the gripper in a reasonable configuration.
            #
            # For reaching, randomizing the gripper opening is not useful.
            # --------------------------------------------------------
            if self.num_actuators >= 6:

                gripper_qpos_index = (
                    self.actuator_qpos_indices[-1]
                )

                sample_qpos[
                    gripper_qpos_index
                ] = 0.5

            self.data.qpos[:] = sample_qpos
            self.data.qvel[:] = 0.0

            mujoco.mj_forward(
                self.model,
                self.data,
            )

            candidate = (
                self._get_end_effector_position()
            )

            # Reject positions below/inside the floor.
            if candidate[2] >= min_height:

                reachable_position = candidate.copy()
                break

        # Restore simulator state.
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # Fallback.
        if reachable_position is None:

            reachable_position = np.array(
                [0.15, 0.0, 0.15],
                dtype=np.float64,
            )

        return reachable_position

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # ------------------------------------------------------------
        # Start robot near neutral position.
        # ------------------------------------------------------------
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0

        # ------------------------------------------------------------
        # Generate reachable target.
        # ------------------------------------------------------------
        target_position = (
            self._sample_reachable_target()
        )

        # ------------------------------------------------------------
        # Reset again because target sampling temporarily moved the arm.
        # ------------------------------------------------------------
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0

        # ------------------------------------------------------------
        # Move mocap target.
        # ------------------------------------------------------------
        if self.model.nmocap > 0:

            target_mocap_id = (
                self.model.body_mocapid[
                    self.target_body_id
                ]
            )

            if target_mocap_id >= 0:

                self.data.mocap_pos[
                    target_mocap_id
                ] = target_position

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.steps = 0

        observation = self._get_obs()

        info = {
            "target_position": target_position.copy(),
            "end_effector_position": (
                self._get_end_effector_position()
            ),
        }

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # Safety check.
        # ------------------------------------------------------------
        if action.shape != self.action_space.shape:

            raise ValueError(
                f"Expected action shape "
                f"{self.action_space.shape}, "
                f"but received {action.shape}."
            )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        previous_end_effector_position = (
            self._get_end_effector_position()
        )

        target_position = (
            self._get_target_position()
        )

        previous_distance = float(
            np.linalg.norm(
                previous_end_effector_position
                - target_position
            )
        )

        # ------------------------------------------------------------
        # Apply desired actuator positions.
        # ------------------------------------------------------------
        self.data.ctrl[:] = action

        mujoco.mj_step(
            self.model,
            self.data,
        )

        self.steps += 1

        # ------------------------------------------------------------
        # Calculate new distance.
        # ------------------------------------------------------------
        end_effector_position = (
            self._get_end_effector_position()
        )

        target_position = (
            self._get_target_position()
        )

        distance = float(
            np.linalg.norm(
                end_effector_position
                - target_position
            )
        )

        # ------------------------------------------------------------
        # Reward
        #
        # 1. Distance penalty.
        # 2. Progress reward.
        # 3. Small action penalty.
        # 4. Large success reward.
        # ------------------------------------------------------------
        progress = (
            previous_distance - distance
        )

        reward = (
            -distance * 1.0
            + progress * 20.0
        )

        # Penalize unnecessarily large commands.
        normalized_action = (
            action - self.action_space.low
        ) / (
            self.action_space.high
            - self.action_space.low
            + 1e-8
        )

        reward -= (
            0.001
            * float(
                np.sum(
                    np.square(
                        normalized_action - 0.5
                    )
                )
            )
        )

        # ------------------------------------------------------------
        # Success.
        # ------------------------------------------------------------
        terminated = (
            distance < self.success_threshold
        )

        if terminated:
            reward += 100.0

        truncated = (
            self.steps >= self.max_steps
        )

        observation = self._get_obs()

        info = {
            "distance": distance,
            "progress": progress,
            "end_effector_position": (
                end_effector_position.copy()
            ),
            "target_position": (
                target_position.copy()
            ),
            "steps": self.steps,
            "success": terminated,
        }

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ================================================================
    # RENDER
    # ================================================================

    def render(self):

        if self.viewer is None:

            self.viewer = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
            )

        self.viewer.sync()

    # ================================================================
    # CLOSE
    # ================================================================

    def close(self):

        if self.viewer is not None:

            self.viewer.close()
            self.viewer = None

        # Remove generated temporary MJCF.
        if (
            hasattr(self, "_temporary_mjcf_path")
            and os.path.exists(
                self._temporary_mjcf_path
            )
        ):

            try:
                os.remove(
                    self._temporary_mjcf_path
                )

            except OSError:
                pass


# ====================================================================
# TEST
# ====================================================================

if __name__ == "__main__":

    env = ArduinobotEnv(
        render_mode="human"
    )

    observation, info = env.reset()

    print("=" * 60)
    print("MuJoCo ArduinoBot Environment Test")
    print("=" * 60)

    print(
        f"Number of actuators: "
        f"{env.num_actuators}"
    )

    print(
        f"Actuators: "
        f"{env.actuator_names}"
    )

    print(
        f"Action shape: "
        f"{env.action_space.shape}"
    )

    print(
        f"Observation shape: "
        f"{env.observation_space.shape}"
    )

    print(
        f"Initial target: "
        f"{info['target_position']}"
    )

    print("=" * 60)

    try:

        for episode in range(10):

            observation, info = env.reset()

            print(
                f"\nEpisode {episode + 1}"
            )

            for step in range(
                env.max_steps
            ):

                # Random action for validation.
                action = (
                    env.action_space.sample()
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                if step % 25 == 0:

                    print(
                        f"Step {step:03d} | "
                        f"Distance: "
                        f"{info['distance']:.4f} m | "
                        f"Reward: "
                        f"{reward:.3f}"
                    )

                time.sleep(0.02)

                if terminated or truncated:

                    status = (
                        "SUCCESS"
                        if terminated
                        else "TIMEOUT"
                    )

                    print(
                        f"Episode ended: "
                        f"{status}"
                    )

                    break

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    finally:

        env.close()
