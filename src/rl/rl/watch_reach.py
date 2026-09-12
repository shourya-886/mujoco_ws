import os
import time

from stable_baselines3 import PPO

from arduinobot_env import ArduinobotEnv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../../ppo_arduinobot_reach.zip")
)


if __name__ == "__main__":
    print("=" * 60)
    print("ArduinoBot PPO Reaching - Policy Evaluation")
    print("=" * 60)
    print(f"Loading model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained PPO model not found:\n{MODEL_PATH}"
        )

    env = ArduinobotEnv(render_mode="human")
    model = PPO.load(MODEL_PATH)

    print("Model loaded successfully.")
    print("Starting evaluation...\n")

    obs, info = env.reset()

    episode = 0
    successes = 0
    timeouts = 0

    episode_min_distance = float("inf")

    for step in range(5000):

        action, _states = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        distance = info.get("distance", float("inf"))
        episode_min_distance = min(
            episode_min_distance,
            distance
        )

        time.sleep(0.02)

        if terminated or truncated:

            episode += 1

            if terminated:
                successes += 1
                result = "SUCCESS"
            else:
                timeouts += 1
                result = "TIMEOUT"

            print(
                f"Episode {episode:3d}: "
                f"{result:7s} | "
                f"final_distance = {distance:.3f} m | "
                f"min_distance = {episode_min_distance:.3f} m"
            )

            episode_min_distance = float("inf")

            obs, info = env.reset()

    env.close()

    total_episodes = successes + timeouts

    print("\n" + "=" * 60)
    print("Evaluation complete")
    print("=" * 60)
    print(f"Episodes : {total_episodes}")
    print(f"Successes: {successes}")
    print(f"Timeouts : {timeouts}")

    if total_episodes > 0:
        success_rate = 100.0 * successes / total_episodes
        print(f"Success rate: {success_rate:.1f}%")