import gymnasium as gym
import numpy as np
import torch

from config import TD3Config
from td3raw import TD3Agent


def run_policy(checkpoint_path, episodes=3):
    cfg = TD3Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg.env_id, render_mode="human")

    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(obs_dim, act_dim, act_limit, cfg, device)
    agent.load(checkpoint_path)

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            action = agent.act(obs, noise_scale=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_return += reward
            ep_len += 1

        print(f"episode={ep} return={ep_return:.2f} len={ep_len}")

    env.close()


if __name__ == "__main__":
    run_policy("checkpoints/td3_walker2d.pt", episodes=3)