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
    def __init__(self, obs_dim, act_dim, act_limit, cfg: TD3Config, device):
        self.cfg = cfg
        self.device = device
        self.act_limit = act_limit

        self.actor = Actor(obs_dim, act_dim, cfg.hidden_dim, act_limit).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.q1 = Critic(obs_dim, act_dim, cfg.hidden_dim).to(device)
        self.q2 = Critic(obs_dim, act_dim, cfg.hidden_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        self.total_updates = 0

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "total_updates": self.total_updates,
        }, path)
    

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])

        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])

        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.q1_opt.load_state_dict(checkpoint["q1_opt"])
        self.q2_opt.load_state_dict(checkpoint["q2_opt"])

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

            target_q1 = self.q1_target(next_obs, next_act)
            target_q2 = self.q2_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2)
            backup = rews + self.cfg.gamma * (1.0 - done) * target_q

        q1_loss = F.mse_loss(self.q1(obs, acts), backup)
        q2_loss = F.mse_loss(self.q2(obs, acts), backup)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        info = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
        }

        if self.total_updates % self.cfg.policy_delay == 0:
            actor_loss = -self.q1(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            with torch.no_grad():
                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)

                for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)

                for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau)
                    p_targ.data.add_(self.cfg.tau * p.data)

            info["actor_loss"] = actor_loss.item()

        self.total_updates += 1
        return info
    
    


@torch.no_grad()
def evaluate(agent, env_id, seed, episodes, device):
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
