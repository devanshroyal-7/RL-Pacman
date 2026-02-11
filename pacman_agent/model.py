from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import CnnPolicy


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(x)
        out = self.conv1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        return x + out


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res(x)
        return x


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.seq1 = ConvSequence(n_input_channels, 32)
        self.seq2 = ConvSequence(32, 64)
        self.seq3 = ConvSequence(64, 64)

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, *observation_space.shape[1:])
            sample = self.seq3(self.seq2(self.seq1(sample)))
            sample = torch.relu(sample)
            flattened_size = sample.numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.max() > 1.0:
            observations = observations.float() / 255.0
        x = self.seq1(observations)
        x = self.seq2(x)
        x = self.seq3(x)
        x = torch.relu(x)
        return self.fc(x)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def build_ppo(
    env,
    tensorboard_log: str | Path,
    device: str = "cuda:0",
    learning_rate: float = 2.5e-4,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    clip_range: float = 0.1,
    max_grad_norm: float = 0.5,
    seed: int = 42,
) -> PPO:
    policy_kwargs: dict[str, Any] = {
        "features_extractor_class": ImpalaCNN,
        "features_extractor_kwargs": {"features_dim": 512},
    }

    model = PPO(
        policy=CnnPolicy,
        env=env,
        learning_rate=linear_schedule(learning_rate),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip_range,
        max_grad_norm=max_grad_norm,
        tensorboard_log=str(tensorboard_log),
        device=device,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )
    return model
