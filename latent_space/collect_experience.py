"""
Script to collect pre-processed 84x84 grayscale experience data from MsPacman.
"""

from typing import Any

import sys
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "pacman_agent"))

from wrappers import make_atari_pacman_env
from dqn_model import build_q_network

# -- Configuration --
NUM_EPISODES = 100
OUTPUT_PATH = "pacman_experience.pkl"
POLICY_PATH = base_dir / "best_policy.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 84
MAX_STEPS_PER_EPISODE = 10000

def prepare_policy_input(obs: np.ndarray) -> torch.Tensor:
    """
    Prepare observation for policy network input. 

    Args: 
        obs: Observation with shape (84, 84, k) where k is the number of frames

    Returns:
        Tensor with shape (1, k, 84, 84)
    """
    state_tensor = torch.from_numpy(obs).unsqueeze(0)           # Add batch dimension
    state_tensor = state_tensor.permute(0, 3, 1, 2).contiguous().float()    # (1, C, H, W)
    return state_tensor


def collect_experience():
    """
    Runs a trained policy agent to collect state transitions from complete episodes.
    The policy uses stacked frames, but we extract single frames for encoder training.
    """
    print(f"Using device: {DEVICE}")

    # Load the trained policy
    policy_path = Path(POLICY_PATH)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {POLICY_PATH}")
    
    print(f"Loading policy from {policy_path}")
    policy_state = torch.load(policy_path, map_location = DEVICE, weights_only = False)
    policy_net = build_q_network(num_actions = 9, frame_stack = 4).to(DEVICE)

    if isinstance(policy_state, dict) and "model" in policy_state:
        policy_net.load_state_dict(policy_state["model"])
    else:
        policy_net.load_state_dict(policy_state)
    policy_net.eval()

    env = make_atari_pacman_env(render_mode=None, frame_stack=4, clip_rewards=False)
    
    experience_buffer = []

    print(f"Collecting experience from {NUM_EPISODES} complete episodes")
   
    for episode in tqdm(range(NUM_EPISODES), desc="Episodes"):
        obs, info = env.reset()
        done = False
        step = 0

        # Extract initial single frame
        state_frame = obs[:, :, -1].copy()

        while not done and step < MAX_STEPS_PER_EPISODE:
            state_tensor = prepare_policy_input(obs).to(DEVICE)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim = 1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state_frame = next_obs[:, :, -1].copy()

            experience_buffer.append((state_frame.copy(), next_state_frame.copy(), action))
            
            obs = next_obs
            state_frame = next_state_frame
            step += 1

    env.close()

    print(f"\nCollected {len(experience_buffer)} state transitions from {NUM_EPISODES} episodes")
    print(f"Average {len(experience_buffer) / NUM_EPISODES:.1f} steps per episode")
    
    print(f"\nSaving experience to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(experience_buffer, f)
    
    print("Experience collection complete.")

if __name__ == "__main__":
    collect_experience()

