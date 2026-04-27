import argparse
import os
import re

import gymnasium as gym
import torch

from config import TD3Config
from td3raw import TD3Agent


CHECKPOINT_PATTERN = re.compile(
    r"^nd3_(?P<env>.+?)_q(?P<num_qs>\d+)_(?P<aggregation>[A-Za-z0-9_]+)_seed(?P<seed>-?\d+)_(?P<stamp>\d{8}_\d{6})\.pt$"
)


def infer_from_checkpoint_name(checkpoint_path):
    name = os.path.basename(checkpoint_path)
    match = CHECKPOINT_PATTERN.match(name)
    if not match:
        return {}
    return match.groupdict()


def run_policy(checkpoint_path, episodes=3, env_id=None):
    cfg = TD3Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_metadata = infer_from_checkpoint_name(checkpoint_path)
    selected_env_id = env_id or name_metadata.get("env") or cfg.env_id
    selected_aggregation = name_metadata.get("aggregation", "min")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_qs_in_checkpoint = len(checkpoint["q_networks"])

    env = gym.make(selected_env_id, render_mode="human")

    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(
        obs_dim,
        act_dim,
        act_limit,
        cfg,
        device,
        num_critics=num_qs_in_checkpoint,
        aggregation_function=selected_aggregation,
    )
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained TD3 policy with rendering."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/nd3_Walker2d-v5_q2_min_seed0_20260419_154742.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to render (default: 3)",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Optional environment override; otherwise inferred from checkpoint name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_policy(args.checkpoint, episodes=args.episodes, env_id=args.env_id)