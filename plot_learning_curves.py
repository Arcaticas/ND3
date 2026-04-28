import argparse
import csv
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt


FILENAME_PATTERN = re.compile(
    r"^nd3_(?P<env>.+?)_q(?P<num_qs>\d+)_(?P<aggregation_function>[A-Za-z0-9_]+)_seed(?P<seed>-?\d+)_(?P<stamp>\d{8}_\d{6})\.csv$"
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
        if metadata["num_qs"] in {"1", "2"}:
            # For 1 or 2 critics, the aggregation function is not relevant, so we can ignore it in the label
            label = f"{metadata['num_qs']} critics"
        else:
            label = f"{metadata['num_qs']} critics - {metadata['aggregation_function']} aggregation"
        seed = metadata["seed"]

        csv_path = os.path.join(log_dir, name)
        steps, values = read_curve(csv_path)
        grouped[env][label].append(
            {
                "seed": seed,
                "path": csv_path,
                "aggregation": metadata["aggregation_function"],
                "steps": steps,
                "values": values,
                "num_qs": int(metadata["num_qs"]),
            }
        )

    return grouped


def parse_aggregation_filter(value):
    if value == "all":
        return None

    selected = {item.strip() for item in value.split(",") if item.strip()}
    if not selected:
        raise argparse.ArgumentTypeError(
            "Aggregation mode cannot be empty. Expected all, min, median, or a comma-separated combination like min,median"
        )

    valid = {"min", "median"}
    invalid = sorted(selected - valid)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid aggregation mode(s): {', '.join(invalid)}. Expected one or more of: all, min, median"
        )
    return selected


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


def make_plots(grouped_runs, output_dir, aggregation_filter=None):
    os.makedirs(output_dir, exist_ok=True)

    if not grouped_runs:
        print("No matching CSV files found. Expected names like nd3_<env>_q<num>_<selector>_seed<seed>_<timestamp>.csv")
        return

    for env, label_runs in grouped_runs.items():
        plt.figure(figsize=(10, 6))
        plotted_any = False

        for label in sorted(label_runs.keys()):
            aggregation = label_runs[label][0]["aggregation"]

            if label_runs[label][0]["num_qs"] > 2: # Check for runs with only 1 or 2 critics since the aggregation function is not relevant for those cases
                if aggregation_filter is not None and aggregation not in aggregation_filter: # Apply aggregation filter if specified, but only for runs with more than 2 critics
                    continue

            runs = label_runs[label]
            steps, means, stds = aggregate_by_step(runs)
            plt.plot(steps, means, label=label)
            plotted_any = True

            if any(s > 0 for s in stds):
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                plt.fill_between(steps, lower, upper, alpha=0.2)

        if not plotted_any:
            plt.close()
            continue

        env_name = env.replace("_", "-")
        plt.title(f"{env_name} Learning Curves - {','.join(aggregation_filter) if aggregation_filter else 'all'} aggregation function")
        plt.xlabel("Environment Steps")
        plt.ylabel("Evaluation Return")
        plt.legend(title="Configuration")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{env}_learning_curves_{','.join(aggregation_filter) if aggregation_filter else 'all'}.png")
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
    parser.add_argument(
        "--aggregation",
        default="all",
        type=parse_aggregation_filter,
        help=(
            "Aggregation mode(s) to plot from the CSV filenames. "
            "Use all, min, median, or a comma-separated combination like min,median. "
            "Default: all"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = collect_runs(args.log_dir)
    make_plots(grouped, args.output_dir, args.aggregation)


if __name__ == "__main__":
    main()
