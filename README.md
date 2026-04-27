# TD3 on Walker2d (Gymnasium + PyTorch)

Minimal implementation of **Twin Delayed DDPG (TD3)** for continuous control on `Walker2d-v5` using Gymnasium and PyTorch.

## Repository Contents

- `train_td3.py`: Main training loop, periodic evaluation, checkpoint saving.
- `play_td3.py`: Loads a trained checkpoint and runs evaluation episodes with rendering.
- `analyze_q_bias.py`: Compares critic Q estimates with realized discounted returns to inspect over/underestimation.
- `td3raw.py`: TD3 components (actor/critic networks, replay buffer, update logic, checkpoint I/O).
- `plot_learning_curves.py`: Builds learning-curve plots from per-run CSV logs.
- `config.py`: Centralized hyperparameter dataclass (`TD3Config`).
- `sanity.py`: Quick environment sanity test for `Walker2d-v5`.
- `environment.yml`: Conda environment definition.
- `README.txt`: Original short setup notes.

## Environment Setup

### 1. Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate nd3-env
```

### 2. Optional fallback install (if dependency resolution breaks)

```bash
pip install torch numpy gymnasium[mujoco]
```

## Quick Sanity Check

Run this first to verify MuJoCo + Gymnasium are working:

```bash
python sanity.py
```

Expected output includes:

```text
Environment working!
```

## Train

```bash
python train_td3.py
```

What training does:

- Uses defaults from `TD3Config` in `config.py`.
- Trains on `Walker2d-v5` for `1_000_000` environment steps by default.
- Prints training episode return/length.
- Runs periodic evaluation every `eval_interval` steps.
- Writes one CSV per run in `logs/` with columns: `step`, `eval_return`.
- Saves checkpoint to:

```text
checkpoints/td3_walker2d.pt
```

CSV filenames include environment, number of critics, aggregation mode, seed, and timestamp:

```text
logs/td3_<env>_q<num_qs>_<selecting_function>_seed<seed>_<yyyymmdd_hhmmss>.csv
```

Example:

```text
logs/td3_Walker2d_v5_q5_min_seed0_20260419_173002.csv
```

## Plot Learning Curves

Generate plots from CSV logs:

```bash
python plot_learning_curves.py
```

Optional output/log directories:

```bash
python plot_learning_curves.py --log-dir logs --output-dir plots
```

Plot behavior:

- Creates one plot per environment.
- Uses labels in the form `q<num_qs>_<selecting_function>`.
- If multiple seeds exist for the same label, plots the mean curve with a shaded ±1 std band.
- Uses environment steps on the x-axis and evaluation return on the y-axis.

## Run a Trained Policy

```bash
python play_td3.py --checkpoint checkpoints/nd3_Hopper-v5_q5_min_seed0_20260425_185049.pt --episodes 3
```

Environment and number of critics are inferred from the checkpoint filename when possible.

## Analyze Q Over/Underestimation

```bash
python analyze_q_bias.py --checkpoint checkpoints/nd3_Hopper-v5_q5_min_seed0_20260425_185049.pt --episodes 10
```

Optional detailed CSV export:

```bash
python analyze_q_bias.py --checkpoint nd3_Hopper-v5_q5_min_seed0_20260425_185049.pt --episodes 10 --output-csv logs/q_bias_hopper_q5.csv
```

The script prints bias statistics where:

```text
bias = predicted Q(s, a) - realized discounted return-to-go
```

Positive mean bias indicates overestimation, negative mean bias indicates underestimation.

## Key Hyperparameters

Configured in `config.py` (`TD3Config`):

- `env_id`: `Walker2d-v5`
- `total_steps`: `1_000_000`
- `buffer_size`: `1_000_000`
- `batch_size`: `256`
- `gamma`: `0.99`
- `tau`: `0.005`
- `actor_lr`: `3e-4`
- `critic_lr`: `3e-4`
- `hidden_dim`: `256`
- `start_steps`: `25_000`
- `policy_noise`: `0.2`
- `noise_clip`: `0.5`
- `policy_delay`: `2`

Edit `config.py` to change experiment settings.

## Notes

- Checkpoint directory is auto-created by `train_td3.py`.
- Device selection is automatic (`cuda` if available, otherwise CPU).
- `environment.yml` currently includes `cpuonly`; remove that entry if you want to use GPU-enabled PyTorch via conda.

## Troubleshooting

- If `Walker2d-v5` fails to load, verify MuJoCo install and that Gymnasium extras are installed: `gymnasium[mujoco]`.
- If rendering does not appear in `play_td3.py`, confirm your local display/GUI support is available.
- If loading fails in `play_td3.py`, ensure `checkpoints/td3_walker2d.pt` exists (run training first).
