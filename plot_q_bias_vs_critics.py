import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from analyze_q_bias import analyze_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot how mean bias changes with critic count, separately for min and median aggregation."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Directory containing checkpoint .pt files (default: checkpoints)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where plots will be written (default: plots)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Evaluation episodes per checkpoint (default: 10)",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Optional environment filter, e.g. Hopper-v5 or HalfCheetah-v5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for all checkpoints",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional discount override used for return-to-go",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap per episode to speed up analysis",
    )
    return parser.parse_args()


def collect_checkpoints(checkpoints_dir, env_id=None):
    names = []
    for name in os.listdir(checkpoints_dir):
        if not name.endswith(".pt"):
            continue
        if env_id is not None and f"nd3_{env_id}_" not in name:
            continue
        names.append(os.path.join(checkpoints_dir, name))
    return sorted(names)


def aggregate_points(point_map):
    return {
        x: sum(values) / len(values)
        for x, values in sorted(point_map.items())
        if values
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoints = collect_checkpoints(args.checkpoints_dir, args.env_id)
    if not checkpoints:
        print("No checkpoints found to plot.")
        return

    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for checkpoint_path in checkpoints:
        result = analyze_checkpoint(
            checkpoint_path,
            episodes=args.episodes,
            seed=args.seed,
            gamma=args.gamma,
            max_steps=args.max_steps,
            print_summary=False,
        )

        env_name = result["env_id"]
        num_qs = result["num_qs"]
        aggregation = result["aggregation"]
        primary_mean_bias = result["primary_mean_bias"]

        if num_qs <= 2:
            grouped[env_name]["min"][num_qs].append(primary_mean_bias)
            grouped[env_name]["median"][num_qs].append(primary_mean_bias)
        else:
            grouped[env_name][aggregation][num_qs].append(primary_mean_bias)

    for env_name, agg_groups in grouped.items():
        plt.figure(figsize=(10, 6))

        for aggregation, style in [("min", "-o"), ("median", "-s")]:
            points = aggregate_points(agg_groups.get(aggregation, {}))
            if not points:
                continue
            xs = sorted(points.keys())
            ys = [points[x] for x in xs]
            plt.plot(xs, ys, style, linewidth=2, markersize=6, label=f"{aggregation} aggregation function")

        plt.axhline(0.0, color="black", linewidth=1, alpha=0.35)
        plt.title(f"{env_name} mean Q bias vs critic count")
        plt.xlabel("Number of critics")
        plt.ylabel("Mean bias (Q - return-to-go)")
        plt.xticks(sorted({x for agg in agg_groups.values() for x in agg.keys()}))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"nd3_{env_name}_q_bias_vs_critics.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()