import copy
import random
from dataclasses import dataclass
from config import TD3Config

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.tensor(self.obs[idxs], device=device),
            acts=torch.tensor(self.acts[idxs], device=device),
            rews=torch.tensor(self.rews[idxs], device=device),
            next_obs=torch.tensor(self.next_obs[idxs], device=device),
            done=torch.tensor(self.done[idxs], device=device),
        )
        return batch


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, act_limit):
        super().__init__()
        self.net = mlp([obs_dim, hidden_dim, hidden_dim, act_dim], nn.ReLU, nn.Identity)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * torch.tanh(self.net(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, hidden_dim, hidden_dim, 1], nn.ReLU, nn.Identity)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, cfg: TD3Config, device, num_critics=2, selecting_function="min"):
        self.cfg = cfg
        self.device = device
        self.act_limit = act_limit
        self.num_critics = num_critics
        self.selecting_function = selecting_function

        self.actor = Actor(obs_dim, act_dim, cfg.hidden_dim, act_limit).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_networks = []
        self.q_targets = []
        for _ in range(self.num_critics):
            q = Critic(obs_dim, act_dim, cfg.hidden_dim).to(device)
            self.q_networks.append(q)
            q_target = copy.deepcopy(q)
            self.q_targets.append(q_target)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opts = [torch.optim.Adam(q.parameters(), lr=cfg.critic_lr) for q in self.q_networks]

        self.total_updates = 0

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q_networks": [q.state_dict() for q in self.q_networks],
            "actor_target": self.actor_target.state_dict(),
            "q_targets": [q_target.state_dict() for q_target in self.q_targets],
            "actor_opt": self.actor_opt.state_dict(),
            "q_opts": [q_opt.state_dict() for q_opt in self.q_opts],
            "total_updates": self.total_updates,
        }, path)
    

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        for i, q in enumerate(self.q_networks):
            q.load_state_dict(checkpoint["q_networks"][i])

        self.actor_target.load_state_dict(checkpoint["actor_target"])
        for i, q_target in enumerate(self.q_targets):
            q_target.load_state_dict(checkpoint["q_targets"][i])

        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        for i, q_opt in enumerate(self.q_opts):
            q_opt.load_state_dict(checkpoint["q_opts"][i])

        self.total_updates = checkpoint["total_updates"]

    @torch.no_grad()
    def act(self, obs, noise_scale=0.0):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act = self.actor(obs_t).cpu().numpy()[0]
        if noise_scale > 0.0:
            act += noise_scale * np.random.randn(*act.shape)
        return np.clip(act, -self.act_limit, self.act_limit)

    def update(self, batch):
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            noise = torch.randn_like(acts) * self.cfg.policy_noise
            noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)

            next_act = self.actor_target(next_obs) + noise
            next_act = torch.clamp(next_act, -self.act_limit, self.act_limit)

            target_qs = []
            for q_target in self.q_targets:
                target_qs.append(q_target(next_obs, next_act))
            if self.selecting_function == "min":
                target_q = torch.min(torch.stack(target_qs), dim=0)[0]
            elif self.selecting_function == "median":
                target_q = torch.median(torch.stack(target_qs), dim=0)[0]
            backup = rews + self.cfg.gamma * (1.0 - done) * target_q

        q_losses = []
        for q, q_opt in zip(self.q_networks, self.q_opts):
            q_loss = F.mse_loss(q(obs, acts), backup)
            q_losses.append(q_loss)

        for q_loss, q_opt in zip(q_losses, self.q_opts):
            q_opt.zero_grad()
            q_loss.backward()
        for q_opt in self.q_opts: # Should we step after all backward() calls to avoid potential issues with shared parameters?
            q_opt.step()

        info = {
            "q_losses": [q_loss.item() for q_loss in q_losses],
        }

        if self.total_updates % self.cfg.policy_delay == 0: # Delayed policy updates
            if self.selecting_function == "min":
                # Use the minimum Q-value from the critics to update the actor, as per TD3's policy update rule
                actor_loss = -torch.min(torch.stack([q(obs, self.actor(obs)) for q in self.q_networks]), dim=0)[0].mean()
            elif self.selecting_function == "median":
                # Use the median Q-value from the critics to update the actor, as per TD3's policy update rule
                actor_loss = -torch.median(torch.stack([q(obs, self.actor(obs)) for q in self.q_networks]), dim=0)[0].mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            with torch.no_grad():
                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)

                for q, q_target in zip(self.q_networks, self.q_targets):
                    for p, p_targ in zip(q.parameters(), q_target.parameters()):
                        p_targ.data.mul_(1 - self.cfg.tau)
                        p_targ.data.add_(self.cfg.tau * p.data)

            info["actor_loss"] = actor_loss.item()

        self.total_updates += 1
        return info
    
    
@torch.no_grad()
def evaluate(agent, env_id, seed, episodes, device): # Evaluate average performance over "episode" amount of episodes, using the current policy without exploration noise. This is typically done every few thousand steps during training to track progress.
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.act(obs, noise_scale=0.0)
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_ret += rew
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))
