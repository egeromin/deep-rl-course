import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    # First, we create our environment called LunarLander-v2
    n_envs = 16
    env = make_vec_env("LunarLander-v3", n_envs=n_envs)

    # Then we reset this environment
    # env.reset()
    #
    # for _ in range(20):
    #     # Take a random action
    #     action = env.action_space.sample()
    #     print("Action taken:", action)
    #
    #     # Do this action in the environment and get
    #     # next_state, reward, terminated, truncated and info
    #     x = env.step(np.array([action] * n_envs))
    #     print(x)
    #
    #     # # If the game is terminated (in our case we land, crashed) or truncated (timeout)
    #     # if terminated or truncated:
    #     #     # Reset the environment
    #     #     print("Environment is reset")
    #     #     env.reset()

    # env.reset()
    # print("_____OBSERVATION SPACE_____ \n")
    # print("Observation Space Shape", env.observation_space.shape)
    # print("Sample observation", env.observation_space.sample())  # Get a random observation

    env.reset()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        # device="mps"  # apple GPU
    )
    model.learn(total_timesteps=1000000)
    model_name = "ppo-LunarLander-v3"
    model.save(model_name)

    env.close()


if __name__ == "__main__":
    main()
