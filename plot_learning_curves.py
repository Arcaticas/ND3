import argparse
import csv
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt


FILENAME_PATTERN = re.compile(
    r"^nd3_(?P<env>.+?)_q(?P<num_qs>\d+)_(?P<selecting_function>[A-Za-z0-9_]+)_seed(?P<seed>-?\d+)_(?P<stamp>\d{8}_\d{6})\.csv$"
)


def read_curve(csv_path):
    steps = []
    eval_returns = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", "eval_return"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV {csv_path} must contain columns {sorted(required)}; found {reader.fieldnames}"
            )

        for row in reader:
            steps.append(int(row["step"]))
            eval_returns.append(float(row["eval_return"]))

    paired = sorted(zip(steps, eval_returns), key=lambda x: x[0])
    return [p[0] for p in paired], [p[1] for p in paired]


def collect_runs(log_dir):
    grouped = defaultdict(lambda: defaultdict(list))

    for name in os.listdir(log_dir):
        if not name.endswith(".csv"):
            continue

        match = FILENAME_PATTERN.match(name)
        if not match:
            continue

        metadata = match.groupdict()
        env = metadata["env"]
        label = f"q{metadata['num_qs']}_{metadata['selecting_function']}"
        seed = metadata["seed"]

        csv_path = os.path.join(log_dir, name)
        steps, values = read_curve(csv_path)
        grouped[env][label].append(
            {
                "seed": seed,
                "path": csv_path,
                "steps": steps,
                "values": values,
            }
        )

    return grouped


def aggregate_by_step(runs):
    by_step = defaultdict(list)
    for run in runs:
        for step, value in zip(run["steps"], run["values"]):
            by_step[step].append(value)

    steps = sorted(by_step.keys())
    means = [mean(by_step[step]) for step in steps]

    stds = []
    for step in steps:
        vals = by_step[step]
        stds.append(stdev(vals) if len(vals) > 1 else 0.0)

    return steps, means, stds


def make_plots(grouped_runs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not grouped_runs:
        print("No matching CSV files found. Expected names like nd3_<env>_q<num>_<selector>_seed<seed>_<timestamp>.csv")
        return

    for env, label_runs in grouped_runs.items():
        plt.figure(figsize=(10, 6))

        for label in sorted(label_runs.keys()):
            runs = label_runs[label]
            steps, means, stds = aggregate_by_step(runs)
            plt.plot(steps, means, label=label)

            if any(s > 0 for s in stds):
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                plt.fill_between(steps, lower, upper, alpha=0.2)

        env_name = env.replace("_", "-")
        plt.title(env_name)
        plt.xlabel("Environment Steps")
        plt.ylabel("Evaluation Return")
        plt.legend(title="Configuration")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{env}_learning_curves.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved plot: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ND3 learning curves from per-run CSV logs."
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing run CSV files (default: logs)",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save generated plot images (default: plots)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = collect_runs(args.log_dir)
    make_plots(grouped, args.output_dir)


if __name__ == "__main__":
    main()
