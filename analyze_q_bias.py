import argparse
import csv
import os
import re

import gymnasium as gym
import numpy as np
import torch

from config import TD3Config
from td3raw import TD3Agent


CHECKPOINT_PATTERN = re.compile(
    r"^nd3_(?P<env>.+?)_q(?P<num_qs>\d+)_(?P<aggregation>[A-Za-z0-9_]+)_seed(?P<seed>-?\d+)_(?P<stamp>\d{8}_\d{6})\.pt$"
)


def infer_checkpoint_metadata(checkpoint_path):
    name = os.path.basename("checkpoints/" + checkpoint_path)
    match = CHECKPOINT_PATTERN.match(name)
    if not match:
        return {}
    return match.groupdict()


def discounted_returns(rewards, gamma):
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def summarize(label, preds, targets):
    bias = preds - targets
    over_frac = float(np.mean(bias > 0.0)) if len(bias) else float("nan")
    under_frac = float(np.mean(bias < 0.0)) if len(bias) else float("nan")
    print(
        f"{label:>14s} | mean_bias={bias.mean(): .4f} | median_bias={np.median(bias): .4f} "
        f"| std_bias={bias.std(): .4f} | mae={np.mean(np.abs(bias)): .4f} "
        f"| over={over_frac: .3f} | under={under_frac: .3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Q over/underestimation by comparing critic Q(s,a) against "
            "realized discounted returns from rollout trajectories."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Optional env override. By default inferred from checkpoint name or config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override. By default inferred from checkpoint name or config.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount used for return-to-go target. Default is config gamma.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for critic forward passes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap per episode to speed up analysis",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write per-step predictions and targets",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = TD3Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_meta = infer_checkpoint_metadata(args.checkpoint)
    env_id = args.env_id or name_meta.get("env") or cfg.env_id
    seed = args.seed
    if seed is None:
        seed = int(name_meta["seed"]) if "seed" in name_meta else cfg.seed
    gamma = args.gamma if args.gamma is not None else cfg.gamma

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_qs = len(checkpoint["q_networks"])
    aggregation = name_meta.get("aggregation", "min")

    env = gym.make(env_id)
    obs0, _ = env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(
        obs_dim,
        act_dim,
        act_limit,
        cfg,
        device,
        num_critics=num_qs,
        aggregation_function=aggregation,
    )
    agent.load(args.checkpoint)

    all_obs = []
    all_acts = []
    all_rtg = []
    episode_returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_rewards = []
        ep_obs = []
        ep_acts = []

        while not done:
            act = agent.act(obs, noise_scale=0.0)
            next_obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            ep_obs.append(obs.copy())
            ep_acts.append(act.copy())
            ep_rewards.append(float(rew))

            obs = next_obs

            if args.max_steps is not None and len(ep_rewards) >= args.max_steps:
                break

        ep_rtg = discounted_returns(ep_rewards, gamma)
        all_obs.extend(ep_obs)
        all_acts.extend(ep_acts)
        all_rtg.extend(ep_rtg.tolist())
        episode_returns.append(sum(ep_rewards))

    env.close()

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    acts_arr = np.asarray(all_acts, dtype=np.float32)
    rtg_arr = np.asarray(all_rtg, dtype=np.float32)

    print(f"checkpoint={args.checkpoint}")
    print(f"env={env_id} episodes={args.episodes} seed={seed} gamma={gamma}")
    print(f"samples={len(rtg_arr)} critics={num_qs} aggregation={aggregation}")
    print(f"avg_episode_return={np.mean(episode_returns):.2f}")
    print()

    q_preds = []
    with torch.no_grad():
        for i in range(0, len(rtg_arr), args.batch_size):
            j = i + args.batch_size
            obs_t = torch.tensor(obs_arr[i:j], dtype=torch.float32, device=device)
            acts_t = torch.tensor(acts_arr[i:j], dtype=torch.float32, device=device)
            q_chunk = [q(obs_t, acts_t).squeeze(-1).cpu().numpy() for q in agent.q_networks]
            q_preds.append(np.stack(q_chunk, axis=0))

    q_preds = np.concatenate(q_preds, axis=1)

    for idx in range(num_qs):
        summarize(f"critic_{idx}", q_preds[idx], rtg_arr)

    q_min = np.min(q_preds, axis=0)
    q_median = np.median(q_preds, axis=0)
    q_mean = np.mean(q_preds, axis=0)
    summarize("agg_min", q_min, rtg_arr)
    summarize("agg_median", q_median, rtg_arr)
    summarize("agg_mean", q_mean, rtg_arr)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = ["sample", "rtg", "q_min", "q_median", "q_mean"] + [
                f"q_{i}" for i in range(num_qs)
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(rtg_arr)):
                row = {
                    "sample": i,
                    "rtg": float(rtg_arr[i]),
                    "q_min": float(q_min[i]),
                    "q_median": float(q_median[i]),
                    "q_mean": float(q_mean[i]),
                }
                for q_idx in range(num_qs):
                    row[f"q_{q_idx}"] = float(q_preds[q_idx, i])
                writer.writerow(row)
        print()
        print(f"Wrote detailed predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
