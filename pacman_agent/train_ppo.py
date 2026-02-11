from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage 
from model import build_ppo
from wrappers import make_atari_pacman_env


class EpisodeRewardCallback(BaseCallback):
    """Callback to log episode rewards as rollout/ep_rew_mean to TensorBoard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_freq = 10
        
    def _on_step(self) -> bool:
        # Check if any episodes completed
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)
        
        # Log periodically when we have episode data
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_rew = np.mean(self.episode_rewards)
            mean_len = np.mean(self.episode_lengths)
            
            self.logger.record("rollout/ep_rew_mean", mean_rew)
            self.logger.record("rollout/ep_len_mean", mean_len)
            
            # Clear after logging to track moving average
            self.episode_rewards = []
            self.episode_lengths = []
        
        return True


def make_vec_env(n_envs: int, frame_stack: int, seed: int, render_mode: Optional[str]):
    """
    Creates a vectorized environment with n_envs running in parallel.
    """
    def make_env(rank):
        def _init():
            # Use the wrapper function from wrappers.py
            env = make_atari_pacman_env(render_mode=render_mode, frame_stack=frame_stack)
            env = Monitor(env)
            # Ensure each environment has a unique seed
            env.reset(seed=seed + rank)
            return env
        return _init

    # Use SubprocVecEnv for true multiprocessing on Windows
    # 'spawn' is required for Windows compatibility
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='spawn')
    env = VecTransposeImage(env)  # Convert HWC to CHW for CnnPolicy
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Ms. Pacman with Stable-Baselines3.")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000, help="Total timesteps to train.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from (.zip).")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000, help="Steps between checkpoints.")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory.")
    parser.add_argument("--checkpoint-dir", type=str, default="ppo_checkpoints/IMPALA", help="Where to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g., cuda:0 for RTX 3080.")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of stacked frames.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments.")
    return parser.parse_args()


def main():
    args = parse_args()

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {args.device}")
    if torch.cuda.is_available():
        print(f"CUDA available: True | GPU: {torch.cuda.get_device_name(0)} | CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, falling back to CPU.")

    env = make_vec_env(n_envs=args.n_envs, frame_stack=args.frame_stack, seed=args.seed, render_mode=None)

    # Load checkpoint if provided, otherwise create new model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, device=args.device, tensorboard_log=log_dir)
        # Get the number of timesteps already trained
        trained_timesteps = model.num_timesteps
        remaining_timesteps = args.total_timesteps - trained_timesteps
        print(f"Resuming training: {trained_timesteps} timesteps already trained, {remaining_timesteps} remaining.")
        if remaining_timesteps <= 0:
            print(f"Model already trained for {trained_timesteps} timesteps (>= {args.total_timesteps}).")
            return
        reset_num_timesteps = False
    else:
        if args.checkpoint:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Starting fresh training.")
        model = build_ppo(
            env=env,
            tensorboard_log=log_dir,
            device=args.device,
            seed=args.seed,
        )
        remaining_timesteps = args.total_timesteps
        reset_num_timesteps = True

    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=str(ckpt_dir),
        name_prefix="impala_pacman",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    episode_reward_cb = EpisodeRewardCallback()
    callbacks = CallbackList([checkpoint_cb, episode_reward_cb])

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="IMPALA",
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        interrupt_path = ckpt_dir / "impala_pacman_interrupt"
        model.save(interrupt_path)
        print(f"Interrupted. Saved checkpoint to {interrupt_path}.")
        return

    final_path = ckpt_dir / "impala_pacman_final"
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.")


if __name__ == "__main__":
    main()

