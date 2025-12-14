import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import ale_py  # Import ALE to register environments


class NoopResetEnv(gym.Wrapper):
	"""Sample initial no-ops on reset to diversify starting states."""
	def __init__(self, env: gym.Env, noop_max: int = 30):
		super().__init__(env)
		self.noop_max = noop_max
		self.noop_action = 0

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		noops = np.random.randint(1, self.noop_max + 1)
		for _ in range(noops):
			obs, _, terminated, truncated, info = self.env.step(self.noop_action)
			if terminated or truncated:
				obs, info = self.env.reset(**kwargs)
		return obs, info

class EpisodicLifeEnv(gym.Wrapper):
	"""Make loss of life terminal, but only reset on true game over."""
	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.lives = 0
		self.was_real_done = True

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		self.was_real_done = terminated or truncated
		# check current lives, make loss of life terminal
		lives = info.get("lives", self.lives)
		if lives < self.lives and lives > 0:
			terminated = True
		self.lives = lives
		return obs, reward, terminated, truncated, info

	def reset(self, **kwargs):
		if self.was_real_done:
			obs, info = self.env.reset(**kwargs)
		else:
			# take a no-op step to advance from terminal life state
			obs, _, terminated, truncated, info = self.env.step(0)
			if terminated or truncated:
				obs, info = self.env.reset(**kwargs)
		self.lives = info.get("lives", 0)
		return obs, info


class MaxAndSkipEnv(gym.Wrapper):
	"""Return only every `skip`-th frame and take max over last two observations."""
	def __init__(self, env: gym.Env, skip: int = 4):
		super().__init__(env)
		self._obs_buffer = collections.deque(maxlen=2)
		self._skip = skip

	def step(self, action):
		total_reward = 0.0
		terminated = False
		truncated = False
		info = {}
		for _ in range(self._skip):
			obs, reward, t, tr, info = self.env.step(action)
			self._obs_buffer.append(obs)
			total_reward += reward
			terminated = terminated or t
			truncated = truncated or tr
			if t or tr:
				break
		max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[-1]) if len(self._obs_buffer) == 2 else obs
		return max_frame, total_reward, terminated, truncated, info

	def reset(self, **kwargs):
		self._obs_buffer.clear()
		return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
	"""Warp frames to 84x84 grayscale, channel-last (H, W, 1)."""
	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

	def observation(self, obs):
		if obs.ndim == 3 and obs.shape[-1] == 3:
			img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
		else:
			img = obs
		resized = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
		return resized[..., None].astype(np.uint8)


class FrameStack(gym.Wrapper):
	"""Stack k last frames along channel axis (H, W, k)."""
	def __init__(self, env: gym.Env, k: int):
		super().__init__(env)
		self.k = k
		self.frames = collections.deque(maxlen=k)
		low = np.repeat(env.observation_space.low, k, axis=-1)
		high = np.repeat(env.observation_space.high, k, axis=-1)
		self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		for _ in range(self.k):
			self.frames.append(obs)
		return self._get_obs(), info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		self.frames.append(obs)
		return self._get_obs(), reward, terminated, truncated, info

	def _get_obs(self):
		return np.concatenate(list(self.frames), axis=-1)


def make_atari_pacman_env(render_mode: str | None = None, frame_stack: int = 4, clip_rewards: bool = True) -> gym.Env:
	"""Create `ALE/MsPacman-v5` with common Atari preprocessing wrappers."""
	env = gym.make("ALE/MsPacman-v5", render_mode=render_mode)
	env = NoopResetEnv(env)
	env = MaxAndSkipEnv(env, skip=4)
	env = WarpFrame(env)
	env = FrameStack(env, k=frame_stack)
	return env


