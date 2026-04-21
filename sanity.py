import gymnasium as gym
import torch

env = gym.make("Walker2d-v5")
obs, _ = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

print("Environment working!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch working! Device: {device}")