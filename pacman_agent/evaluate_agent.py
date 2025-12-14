#!/usr/bin/env python3
"""
Evaluate a trained DQN agent on Pacman.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from wrappers import make_atari_pacman_env
from dqn_model import build_q_network

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with same architecture
    model = build_q_network(num_actions=9, frame_stack=4).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Loaded checkpoint from frame {checkpoint['frame']}")
    return model

def evaluate_agent(model, env, num_episodes: int = 5, render: bool = True):
    """Evaluate the agent for multiple episodes."""
    device = next(model.parameters()).device
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done and step < 10000:  # Max steps per episode
            # Convert observation to tensor
            state_tensor = torch.from_numpy(obs).unsqueeze(0)  # (1, H, W, C)
            state_tensor = state_tensor.permute(0, 3, 1, 2).contiguous().to(device)  # (1, C, H, W)
            
            # Select action (greedy)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            if render and step % 10 == 0:  # Print progress every 10 steps
                print(f"Step {step}: action={action}, reward={reward}, total_reward={episode_reward:.1f}")
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished: {step} steps, total reward: {episode_reward:.1f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"\nEvaluation complete!")
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.1f}")
    print(f"Rewards: {[f'{r:.1f}' for r in episode_rewards]}")
    
    return episode_rewards

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--save-video", action="store_true", help="Save episode as video")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    model = load_checkpoint(args.checkpoint, device)
    
    # Create environment
    render_mode = "human" if not args.no_render else None
    env = make_atari_pacman_env(render_mode=render_mode, frame_stack=4, clip_rewards=False)
    
    try:
        # Evaluate agent
        episode_rewards = evaluate_agent(
            model, env, 
            num_episodes=args.episodes, 
            render=not args.no_render
        )
        
    finally:
        env.close()

if __name__ == "__main__":
    main()

