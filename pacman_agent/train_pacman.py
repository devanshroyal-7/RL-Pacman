import os
import time
import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2

from wrappers import make_atari_pacman_env
from dqn_model import build_q_network
from replay_buffer import ReplayBuffer


def select_action(eps: float, q_net: nn.Module, state: torch.Tensor, num_actions: int) -> int:
	"""Standard epsilon-greedy action selection."""
	if random.random() < eps:
		return random.randrange(num_actions)
	
	with torch.no_grad():
		q_values = q_net(state)
		return int(q_values.argmax(dim=1).item())


def linear_schedule(start: float, end: float, frac: float) -> float:
	return start + frac * (end - start)


def main():
	# Config - Research-backed hyperparameters for ALE Pacman
	seed = 42
	frame_stack = 4
	buffer_size = 1_000_000  # Large buffer for stable learning
	batch_size = 32  # Optimal batch size for DQN
	learning_rate = 2.5e-4  # RMSProp learning rate (Nature paper standard)
	gamma = 0.99
	train_start = 100_000  # Wait for buffer to fill properly
	target_update_freq = 1_000  # More frequent updates for stability
	max_frames = 2_000_000  # Standard training budget
	eps_start = 1.0
	eps_end = 0.01  # 1% final exploration (lower for better exploitation)
	eps_decay_frames = 1_000_000  # Linear decay over 1M frames
	log_interval = 10_000
	checkpoint_dir = Path("checkpoints")
	checkpoint_dir.mkdir(exist_ok=True)
	
	# TensorBoard logging
	log_dir = Path("runs")
	log_dir.mkdir(exist_ok=True)
	writer = SummaryWriter(log_dir / f"pacman_dqn_{int(time.time())}")
	
	# Episode recording
	record_dir = Path("episode_recordings")
	record_dir.mkdir(exist_ok=True)

	# Seeding
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# Device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	# CUDA diagnostics
	try:
		print(f"CUDA available: {torch.cuda.is_available()}")
		if torch.cuda.is_available():
			print(f"GPU: {torch.cuda.get_device_name(0)}")
			print(f"CUDA version: {getattr(torch.version, 'cuda', 'unknown')}")
			print(f"cuDNN: {torch.backends.cudnn.version()}")
	except Exception as _e:
		pass

	# Env
	env = make_atari_pacman_env(render_mode=None, frame_stack=frame_stack, clip_rewards=False)
	obs, info = env.reset(seed=seed)
	num_actions: int = env.action_space.n
	state_shape = obs.shape  # (84, 84, frame_stack)

	# Networks
	policy_net = build_q_network(num_actions, frame_stack).to(device)
	target_net = build_q_network(num_actions, frame_stack).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate, alpha=0.95, eps=1e-2)  # RMSProp optimizer (Nature paper standard)
	loss_fn = nn.SmoothL1Loss()

	# Replay buffer
	replay = ReplayBuffer(buffer_size, state_shape)

	# Start fresh training with research-backed hyperparameters
	print("Starting training for ALE Pacman...")
	start_frame = 0
	action_counts = {i: 0 for i in range(num_actions)}
	
	# Optional: Load from research checkpoint if available
	latest_ckpt = None
	if checkpoint_dir.exists():
		ckpts = sorted(checkpoint_dir.glob("dqn_pacman_research_*.pt"))
		if ckpts:
			latest_ckpt = ckpts[-1]
			print(f"Found research checkpoint: {latest_ckpt}")
			choice = input("Load from research checkpoint? (y/n): ").lower().strip()
			if choice == 'y':
				try:
					ckpt = torch.load(latest_ckpt, map_location=device)
					policy_net.load_state_dict(ckpt.get("model", {}))
					target_net.load_state_dict(policy_net.state_dict())
					if "optimizer" in ckpt and ckpt["optimizer"]:
						optimizer.load_state_dict(ckpt["optimizer"])
					start_frame = int(ckpt.get("frame", 0))
					action_counts = ckpt.get("action_counts", {i: 0 for i in range(num_actions)})
					print(f"Resumed from {latest_ckpt} at frame={start_frame}")
					if start_frame >= max_frames:
						max_frames = start_frame + 1_000_000
				except Exception as e:
					print(f"Warning: failed to load checkpoint {latest_ckpt}: {e}")
					start_frame = 0
			else:
				print("Starting fresh training...")
				start_frame = 0

	# Preprocess helper: NHWC uint8 -> NCHW float tensor on device
	def to_tensor(state_np: np.ndarray) -> torch.Tensor:
		st = torch.from_numpy(state_np).unsqueeze(0)  # (1, H, W, C)
		st = st.permute(0, 3, 1, 2).contiguous().to(device)  # (1, C, H, W)
		return st

	# Training loop
	frame = start_frame
	episode = 0
	episode_reward = 0.0
	episode_rewards = []
	start_time = time.time()
	
	# Action counting for UCB exploration (already initialized above)
	
	# Episode recording setup
	episode_frames = []
	record_episodes = [100, 500, 1000, 2000, 5000, 10000, 20000]  # More recording points

	state = obs
	while frame < max_frames:
		frac = min(1.0, frame / eps_decay_frames)
		eps = linear_schedule(eps_start, eps_end, frac)
		state_t = to_tensor(state)
		action = select_action(eps, policy_net, state_t, num_actions)
		action_counts[action] += 1

		next_state, reward, terminated, truncated, info = env.step(action)
		done = terminated or truncated
		replay.push(state, action, reward, next_state, done)

		# Record episode frames for visualization
		if episode in record_episodes:
			# Convert stacked frames to single frame for recording
			display_frame = next_state[:, :, -1]  # Take last frame from stack
			episode_frames.append(display_frame)

		state = next_state
		frame += 1
		episode_reward += reward

		# Optimize
		if len(replay) >= train_start:
			states_b, actions_b, rewards_b, next_states_b, dones_b = replay.sample(batch_size, device)
			q_values = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
			with torch.no_grad():
				next_q = target_net(next_states_b).max(1)[0]
				target = rewards_b + gamma * (1.0 - dones_b) * next_q
			loss = loss_fn(q_values, target)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
			optimizer.step()

		# Target net update
		if frame % target_update_freq == 0:
			target_net.load_state_dict(policy_net.state_dict())

		# Logging
		if frame % log_interval == 0 and frame > 0:
			fps = frame / max(1e-6, (time.time() - start_time))
			avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
			reward_std = np.std(episode_rewards[-100:]) if len(episode_rewards) >= 10 else 0
			
			# Calculate action distribution for exploration analysis
			total_actions = sum(action_counts.values())
			action_dist = [action_counts[i] / max(1, total_actions) for i in range(num_actions)]
			exploration_entropy = -sum(p * np.log(p + 1e-8) for p in action_dist if p > 0)
			
			print(f"frame={frame} eps={eps:.3f} replay={len(replay)} reward={episode_reward:.1f} avg_reward={avg_reward:.1f}±{reward_std:.1f} fps={fps:.1f} entropy={exploration_entropy:.3f}")
			
			# TensorBoard logging
			writer.add_scalar("Training/Epsilon", eps, frame)
			writer.add_scalar("Training/FPS", fps, frame)
			writer.add_scalar("Training/ReplayBufferSize", len(replay), frame)
			writer.add_scalar("Training/ExplorationEntropy", exploration_entropy, frame)
			writer.add_scalar("Training/RewardStd", reward_std, frame)
			if episode_rewards:
				writer.add_scalar("Training/AverageReward", avg_reward, frame)
				writer.add_scalar("Training/EpisodeReward", episode_reward, frame)

		if done:
			print(f"Episode {episode} finished, reward={episode_reward:.1f}")
			episode_rewards.append(episode_reward)
			
			# Save episode recording if this was a recorded episode
			if episode in record_episodes and episode_frames:
				video_path = record_dir / f"episode_{episode}.mp4"
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				height, width = episode_frames[0].shape
				video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
				
				for frame_img in episode_frames:
					# Convert grayscale to 3-channel for video
					frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
					video_writer.write(frame_bgr)
				video_writer.release()
				print(f"Saved episode recording to {video_path}")
				episode_frames = []
			
			# TensorBoard episode logging
			writer.add_scalar("Episode/Reward", episode_reward, episode)
			writer.add_scalar("Episode/Length", len(episode_frames) if episode_frames else 0, episode)
			
			episode += 1
			state, info = env.reset()
			episode_reward = 0.0

		# Checkpoint with research standard naming
		if frame % (target_update_freq * 10) == 0 and frame > 0:
			avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
			ckpt_path = checkpoint_dir / f"dqn_pacman_research_{frame}_avg{avg_reward:.1f}.pt"
			torch.save({
				"frame": frame,
				"model": policy_net.state_dict(),
				"optimizer": optimizer.state_dict(),
				"replay_size": len(replay),
				"seed": seed,
				"action_counts": action_counts,
				"avg_reward": avg_reward,
				"eps": eps,
			}, ckpt_path)
			print(f"Saved checkpoint to {ckpt_path}")

	print("Training complete.")
	writer.close()
	env.close()


if __name__ == "__main__":
	main()


