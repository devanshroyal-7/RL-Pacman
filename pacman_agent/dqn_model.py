from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDQN(nn.Module):
	"""Nature DQN-style CNN for stacked grayscale frames.

	Input shape: (B, C, 84, 84) where C is number of stacked frames.
	"""
	def __init__(self, in_channels: int, num_actions: int):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU(inplace=True),
		)
		self.fc = nn.Sequential(
			nn.Linear(64 * 7 * 7, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, num_actions),
		)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.float() / 255.0
		x = self.conv(x)
		x = torch.flatten(x, 1)
		return self.fc(x)


def build_q_network(num_actions: int, frame_stack: int) -> ConvDQN:
	return ConvDQN(in_channels=frame_stack, num_actions=num_actions)


