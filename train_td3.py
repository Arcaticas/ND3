import random
import numpy as np
import torch
import gymnasium as gym
import os

from config import TD3Config
from td3raw import TD3Agent, ReplayBuffer, evaluate

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(cfg: TD3Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg.env_id)
    obs, _ = env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(obs_dim, act_dim, act_limit, cfg, device)
    replay = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)

    episode_return = 0.0
    episode_len = 0

    for t in range(cfg.total_steps):
        if t < cfg.start_steps:
            act = env.action_space.sample()
        else:
            act = agent.act(obs, noise_scale=cfg.exploration_noise)

        next_obs, rew, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        replay.add(obs, act, rew, next_obs, float(done))

        obs = next_obs
        episode_return += rew
        episode_len += 1

        if done:
            print(f"train episode return={episode_return:.2f} len={episode_len}")
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        if t >= cfg.update_after and replay.size >= cfg.batch_size:
            for _ in range(cfg.update_every):
                batch = replay.sample(cfg.batch_size, device)
                agent.update(batch)

        if (t + 1) % cfg.eval_interval == 0:
            avg_ret = evaluate(agent, cfg.env_id, cfg.seed, cfg.eval_episodes, device)
            print(f"step={t+1} eval_return={avg_ret:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/td3_walker2d.pt")
    print("Saved checkpoint to checkpoints/td3_walker2d.pt")

    env.close()


if __name__ == "__main__":
    cfg = TD3Config()
    train(cfg)