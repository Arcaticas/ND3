# TD3 on Walker2d (Gymnasium + PyTorch)

Minimal implementation of **Twin Delayed DDPG (TD3)** for continuous control on `Walker2d-v5` using Gymnasium and PyTorch.

## Repository Contents

- `train_td3.py`: Main training loop, periodic evaluation, checkpoint saving.
- `play_td3.py`: Loads a trained checkpoint and runs evaluation episodes with rendering.
- `td3raw.py`: TD3 components (actor/critic networks, replay buffer, update logic, checkpoint I/O).
- `config.py`: Centralized hyperparameter dataclass (`TD3Config`).
- `sanity.py`: Quick environment sanity test for `Walker2d-v5`.
- `environment.yml`: Conda environment definition.
- `README.txt`: Original short setup notes.

## Environment Setup

### 1. Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate td3-walker2d
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
- Trains on `Walker2d-v5` for `1_000_000` steps by default.
- Prints training episode return/length.
- Runs periodic evaluation every `eval_interval` steps.
- Saves checkpoint to:

```text
checkpoints/td3_walker2d.pt
```

## Run a Trained Policy

```bash
python play_td3.py
```

By default this loads:

```text
checkpoints/td3_walker2d.pt
```

and runs 3 rendered episodes.

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
