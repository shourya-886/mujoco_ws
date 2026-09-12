
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from arduinobot_env import FurutaPendulumEnv


def make_env():
    """
    Create the Furuta pendulum training environment.

    Rendering is disabled during training because rendering significantly
    slows down PPO training.
    """

    env = FurutaPendulumEnv(
        render_mode=None,
        max_steps=1000,
        upright_threshold=0.17,
        fall_threshold=0.8,
        control_substeps=10,
        init_alpha_noise=0.05,
        velocity_penalty_weight=0.01,
        action_penalty_weight=0.001,
    )

    env = Monitor(env)

    return env


if __name__ == "__main__":

    # ================================================================
    # ENVIRONMENT
    # ================================================================

    env = DummyVecEnv([make_env])

    print("=" * 60)
    print("MuJoCo Furuta Pendulum PPO Training")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print("=" * 60)


    # ================================================================
    # PPO MODEL
    # ================================================================

    model = PPO(
        policy="MlpPolicy",
        env=env,

        verbose=1,

        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,

        tensorboard_log="./tb_logs_furuta/",
    )


    # ================================================================
    # TRAINING
    # ================================================================

    TOTAL_TIMESTEPS = 1_000_000

    print()
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )


    # ================================================================
    # SAVE MODEL
    # ================================================================

    save_path = os.path.expanduser(
        "~/mujoco_ws/ppo_furuta_pendulum"
    )

    model.save(save_path)

    print()
    print("=" * 60)
    print("Training complete")
    print(f"Model saved to: {save_path}.zip")
    print("=" * 60)


    env.close()