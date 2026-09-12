import time

from stable_baselines3 import PPO

from arduinobot_env import ArduinobotEnv

MODEL_PATH = "../../../ppo_arduinobot_reach.zip"  # relative to src/rl/rl/


if __name__ == "__main__":
    env = ArduinobotEnv(render_mode="human")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    episode = 0

    for i in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)

        if terminated or truncated:
            episode += 1
            result = "SUCCESS" if terminated else "timeout"
            print(f"episode {episode}: {result}  final_distance={info['distance']:.3f}")
            obs, info = env.reset()

    env.close()