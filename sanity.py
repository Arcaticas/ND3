import gymnasium as gym

env = gym.make("Walker2d-v5")
obs, _ = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

print("Environment working!")