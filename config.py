from dataclasses import dataclass


@dataclass
class TD3Config:
    env_id: str = "Walker2d-v5"
    seed: int = 0
    total_steps: int = 1_000_000

    buffer_size: int = 1_000_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 256

    start_steps: int = 25_000
    update_after: int = 1_000
    update_every: int = 1

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1

    eval_episodes: int = 5
    eval_interval: int = 10_000