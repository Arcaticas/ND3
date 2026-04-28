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
    name = os.path.basename(checkpoint_path)
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


def bias_stats(preds, targets):
    bias = preds - targets
    if len(bias) == 0:
        return {
            "mean_bias": float("nan"),
            "median_bias": float("nan"),
            "std_bias": float("nan"),
            "mae": float("nan"),
            "over_frac": float("nan"),
            "under_frac": float("nan"),
        }

    return {
        "mean_bias": float(np.mean(bias)),
        "median_bias": float(np.median(bias)),
        "std_bias": float(np.std(bias)),
        "mae": float(np.mean(np.abs(bias))),
        "over_frac": float(np.mean(bias > 0.0)),
        "under_frac": float(np.mean(bias < 0.0)),
    }


def summarize(label, preds, targets):
    stats = bias_stats(preds, targets)
    print(
        f"{label:>14s} | mean_bias={stats['mean_bias']: .4f} | median_bias={stats['median_bias']: .4f} "
        f"| std_bias={stats['std_bias']: .4f} | mae={stats['mae']: .4f} "
        f"| over={stats['over_frac']: .3f} | under={stats['under_frac']: .3f}"
    )
    return stats


def analyze_checkpoint(
    checkpoint_path,
    episodes=10,
    env_id=None,
    seed=None,
    gamma=None,
    batch_size=4096,
    max_steps=None,
    device=None,
    print_summary=True,
):
    cfg = TD3Config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_meta = infer_checkpoint_metadata(checkpoint_path)
    resolved_env_id = env_id or name_meta.get("env") or cfg.env_id
    resolved_seed = seed
    if resolved_seed is None:
        resolved_seed = int(name_meta["seed"]) if "seed" in name_meta else cfg.seed
    resolved_gamma = gamma if gamma is not None else cfg.gamma

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_qs = len(checkpoint["q_networks"])
    aggregation = name_meta.get("aggregation", "min")

    env = gym.make(resolved_env_id)
    obs0, _ = env.reset(seed=resolved_seed)
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
    agent.load(checkpoint_path)

    all_obs = []
    all_acts = []
    all_rtg = []
    episode_returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=resolved_seed + ep)
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

            if max_steps is not None and len(ep_rewards) >= max_steps:
                break

        ep_rtg = discounted_returns(ep_rewards, resolved_gamma)
        all_obs.extend(ep_obs)
        all_acts.extend(ep_acts)
        all_rtg.extend(ep_rtg.tolist())
        episode_returns.append(sum(ep_rewards))

    env.close()

    obs_arr = np.asarray(all_obs, dtype=np.float32)
    acts_arr = np.asarray(all_acts, dtype=np.float32)
    rtg_arr = np.asarray(all_rtg, dtype=np.float32)

    q_preds = []
    with torch.no_grad():
        for i in range(0, len(rtg_arr), batch_size):
            j = i + batch_size
            obs_t = torch.tensor(obs_arr[i:j], dtype=torch.float32, device=device)
            acts_t = torch.tensor(acts_arr[i:j], dtype=torch.float32, device=device)
            q_chunk = [q(obs_t, acts_t).squeeze(-1).cpu().numpy() for q in agent.q_networks]
            q_preds.append(np.stack(q_chunk, axis=0))

    q_preds = np.concatenate(q_preds, axis=1) if q_preds else np.empty((num_qs, 0), dtype=np.float32)

    critic_stats = {}
    for idx in range(num_qs):
        critic_stats[f"critic_{idx}"] = bias_stats(q_preds[idx], rtg_arr)

    agg_min = bias_stats(np.min(q_preds, axis=0), rtg_arr)
    agg_median = bias_stats(np.median(q_preds, axis=0), rtg_arr)
    agg_mean = bias_stats(np.mean(q_preds, axis=0), rtg_arr)
    aggregate_stats = {
        "min": agg_min,
        "median": agg_median,
        "mean": agg_mean,
    }
    primary_aggregate = aggregation if aggregation in aggregate_stats else "min"

    result = {
        "checkpoint": checkpoint_path,
        "env_id": resolved_env_id,
        "seed": resolved_seed,
        "gamma": resolved_gamma,
        "num_qs": num_qs,
        "aggregation": aggregation,
        "primary_aggregate": primary_aggregate,
        "primary_mean_bias": aggregate_stats[primary_aggregate]["mean_bias"],
        "episode_returns": episode_returns,
        "avg_episode_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "critic_stats": critic_stats,
        "aggregate_stats": aggregate_stats,
    }

    if print_summary:
        print(f"checkpoint={checkpoint_path}")
        print(f"env={resolved_env_id} episodes={episodes} seed={resolved_seed} gamma={resolved_gamma}")
        print(f"samples={len(rtg_arr)} critics={num_qs} aggregation={aggregation}")
        print(f"avg_episode_return={result['avg_episode_return']:.2f}")
        print()

        for idx in range(num_qs):
            stats = critic_stats[f"critic_{idx}"]
            print(
                f"{f'critic_{idx}':>14s} | mean_bias={stats['mean_bias']: .4f} | median_bias={stats['median_bias']: .4f} "
                f"| std_bias={stats['std_bias']: .4f} | mae={stats['mae']: .4f} "
                f"| over={stats['over_frac']: .3f} | under={stats['under_frac']: .3f}"
            )

        for label in ["min", "median", "mean"]:
            stats = aggregate_stats[label]
            print(
                f"{f'agg_{label}':>14s} | mean_bias={stats['mean_bias']: .4f} | median_bias={stats['median_bias']: .4f} "
                f"| std_bias={stats['std_bias']: .4f} | mae={stats['mae']: .4f} "
                f"| over={stats['over_frac']: .3f} | under={stats['under_frac']: .3f}"
            )

    return result


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
    result = analyze_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        env_id=args.env_id,
        seed=args.seed,
        gamma=args.gamma,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        print_summary=True,
    )

    if args.output_csv:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        num_qs = len(checkpoint["q_networks"])
        cfg = TD3Config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Recompute the detailed sample arrays only when the user explicitly asks for CSV output.
        name_meta = infer_checkpoint_metadata(args.checkpoint)
        env_id = args.env_id or name_meta.get("env") or cfg.env_id
        seed = args.seed
        if seed is None:
            seed = int(name_meta["seed"]) if "seed" in name_meta else cfg.seed
        gamma = args.gamma if args.gamma is not None else cfg.gamma

        env = gym.make(env_id)
        obs0, _ = env.reset(seed=seed)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_limit = float(env.action_space.high[0])

        aggregation = name_meta.get("aggregation", "min")
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

        env.close()

        obs_arr = np.asarray(all_obs, dtype=np.float32)
        acts_arr = np.asarray(all_acts, dtype=np.float32)
        rtg_arr = np.asarray(all_rtg, dtype=np.float32)
        q_preds = []
        with torch.no_grad():
            for i in range(0, len(rtg_arr), args.batch_size):
                j = i + args.batch_size
                obs_t = torch.tensor(obs_arr[i:j], dtype=torch.float32, device=device)
                acts_t = torch.tensor(acts_arr[i:j], dtype=torch.float32, device=device)
                q_chunk = [q(obs_t, acts_t).squeeze(-1).cpu().numpy() for q in agent.q_networks]
                q_preds.append(np.stack(q_chunk, axis=0))

        q_preds = np.concatenate(q_preds, axis=1)
        q_min = np.min(q_preds, axis=0)
        q_median = np.median(q_preds, axis=0)
        q_mean = np.mean(q_preds, axis=0)

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
