from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Transition:
	state: np.ndarray
	action: int
	reward: float
	next_state: np.ndarray
	done: bool


class ReplayBuffer:
	def __init__(self, capacity: int, state_shape: tuple[int, int, int]):
		self.capacity = capacity
		self.position = 0
		self.full = False
		self.state_shape = state_shape
		self.states = np.empty((capacity,) + state_shape, dtype=np.uint8)
		self.next_states = np.empty((capacity,) + state_shape, dtype=np.uint8)
		self.actions = np.empty((capacity,), dtype=np.int64)
		self.rewards = np.empty((capacity,), dtype=np.float32)
		self.dones = np.empty((capacity,), dtype=np.bool_)

	def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
		idx = self.position
		self.states[idx] = state
		self.actions[idx] = action
		self.rewards[idx] = reward
		self.next_states[idx] = next_state
		self.dones[idx] = done
		self.position = (self.position + 1) % self.capacity
		self.full = self.full or self.position == 0

	def __len__(self) -> int:
		return self.capacity if self.full else self.position

	def sample(self, batch_size: int, device: torch.device):
		max_idx = self.capacity if self.full else self.position
		idxs = np.random.randint(0, max_idx, size=batch_size)
		states = torch.from_numpy(self.states[idxs]).to(device)
		actions = torch.from_numpy(self.actions[idxs]).to(device)
		rewards = torch.from_numpy(self.rewards[idxs]).to(device)
		next_states = torch.from_numpy(self.next_states[idxs]).to(device)
		dones = torch.from_numpy(self.dones[idxs].astype(np.float32)).to(device)
		# NHWC -> NCHW
		states = states.permute(0, 3, 1, 2).contiguous()
		next_states = next_states.permute(0, 3, 1, 2).contiguous()
		return states, actions, rewards, next_states, dones



