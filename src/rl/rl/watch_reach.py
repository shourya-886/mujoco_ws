import os
import time

from stable_baselines3 import PPO

from arduinobot_env import FurutaPendulumEnv


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_PATH = os.path.expanduser(
    "~/mujoco_ws/ppo_furuta_pendulum.zip"
)

NUM_EPISODES = 10


# ================================================================
# POLICY EVALUATION
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Furuta Pendulum PPO - Policy Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------
    # Check model
    # ------------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\nTrained model not found:\n{MODEL_PATH}\n"
            "\nTrain the policy first using train_reach.py."
        )

    print(f"Loading model: {MODEL_PATH}")

    # ------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------

    env = FurutaPendulumEnv(
        render_mode="human"
    )

    # ------------------------------------------------------------
    # Load trained PPO policy
    # ------------------------------------------------------------

    model = PPO.load(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print()
    print("Starting evaluation...")
    print("=" * 60)

    try:

        for episode in range(NUM_EPISODES):

            observation, info = env.reset()

            episode_reward = 0.0
            step = 0

            print()
            print(f"Episode {episode + 1}/{NUM_EPISODES}")
            print(
                f"Initial alpha: {info['alpha']:.4f} rad"
            )

            while True:

                # ------------------------------------------------
                # Run trained policy
                # ------------------------------------------------

                action, _states = model.predict(
                    observation,
                    deterministic=True
                )

                # ------------------------------------------------
                # Step environment
                # ------------------------------------------------

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                episode_reward += reward
                step += 1

                # ------------------------------------------------
                # Print status periodically
                # ------------------------------------------------

                if step % 50 == 0:

                    print(
                        f"Step {step:04d} | "
                        f"theta: {info['theta']:+.3f} | "
                        f"alpha: {info['alpha']:+.3f} | "
                        f"reward: {reward:+.4f}"
                    )

                if terminated or truncated:

                    break

                # Small delay so the viewer remains easy to watch.
                time.sleep(0.002)

            # ----------------------------------------------------
            # Episode result
            # ----------------------------------------------------

            if info["fell_over"]:
                status = "FELL"

            elif info["hit_arm_limit"]:
                status = "ARM LIMIT"

            elif truncated:
                status = "TIMEOUT"

            else:
                status = "ENDED"

            print(
                f"Episode finished: {status}"
            )

            print(
                f"Steps:           {step}"
            )

            print(
                f"Episode reward:  {episode_reward:.3f}"
            )

            print(
                f"Final theta:     {info['theta']:+.3f} rad"
            )

            print(
                f"Final alpha:     {info['alpha']:+.3f} rad"
            )

            print(
                f"Upright:         {info['upright']}"
            )

            # Give the viewer a moment before resetting.
            time.sleep(1.0)

    except KeyboardInterrupt:

        print()
        print("Evaluation stopped by user.")

    finally:

        env.close()

        print()
        print("=" * 60)
        print("Evaluation finished")
        print("=" * 60)

