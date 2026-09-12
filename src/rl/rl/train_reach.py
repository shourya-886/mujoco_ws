import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from arduinobot_env import ArduinobotEnv


def make_env():
    # No rendering during training -- much faster, and a display isn't
    # needed until you want to watch the trained policy afterward.
    env = ArduinobotEnv(render_mode=None)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs_reach/",
    )

    model.learn(total_timesteps=500_000)

    save_path = os.path.expanduser("~/mujoco_ws/ppo_arduinobot_reach")
    model.save(save_path)
    print(f"Saved model to {save_path}.zip")