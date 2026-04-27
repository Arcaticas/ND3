import random
import csv
import numpy as np
import torch
import gymnasium as gym
import os
from datetime import datetime
from wakepy import keep

from config import TD3Config
from td3raw import TD3Agent, ReplayBuffer, evaluate

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(cfg: TD3Config, num_qs: int = 2, aggregation_function: str = "min"):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg.env_id)
    obs, _ = env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(obs_dim, act_dim, act_limit, cfg, device, num_qs, aggregation_function)
    replay = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)

    os.makedirs("logs", exist_ok=True)
    env_tag = cfg.env_id.replace("-", "_").replace("/", "_")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(
        "logs",
        f"nd3_{env_tag}_q{num_qs}_{aggregation_function}_seed{cfg.seed}_{run_stamp}.csv",
    )

    episode_return = 0.0
    episode_len = 0

    with open(metrics_path, "w", newline="") as metrics_file: # Open a CSV file to log evaluation metrics during training
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "step",
                "eval_return",
            ],
        )
        metrics_writer.writeheader()

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

            if t >= cfg.update_after and replay.size >= cfg.batch_size: # Start updates after we've collected some data and have enough for a batch
                for _ in range(cfg.update_every):
                    batch = replay.sample(cfg.batch_size, device)
                    agent.update(batch)

            if (t + 1) % cfg.eval_interval == 0: # Evaluate the agent every eval_interval steps during training to track progress
                avg_ret = evaluate(agent, cfg.env_id, cfg.seed, cfg.eval_episodes, device)
                print(f"step={t+1} eval_return={avg_ret:.2f}")
                metrics_writer.writerow(
                    {
                        "step": t + 1,
                        "eval_return": avg_ret,
                    }
                )
                metrics_file.flush()

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_file = f"checkpoints/nd3_{cfg.env_id}_q{num_qs}_{aggregation_function}_seed{cfg.seed}_{run_stamp}.pt"
    agent.save(checkpoint_file)
    print(f"Saved checkpoint to {checkpoint_file}")
    print(f"Saved eval metrics to {metrics_path}")

    env.close()


if __name__ == "__main__":
    cfg = TD3Config()
    # cfg.env_id = "HalfCheetah-v5" # Change environment
    cfg.env_id = "Hopper-v5" # Change environment

    # Prevent system sleep while running the script
    with keep.running():
        train(cfg, num_qs=4, aggregation_function="min") # aggregation_function determines how the Q-values from multiple critics are combined to update the actor. "min" uses the minimum Q-value (standard TD3), while "median" uses the median Q-value, which can be more robust to outliers and may lead to better performance in some cases.
        train(cfg, num_qs=5, aggregation_function="min") # aggregation_function determines how the Q-values from multiple critics are combined to update the actor. "min" uses the minimum Q-value (standard TD3), while "median" uses the median Q-value, which can be more robust to outliers and may lead to better performance in some cases.