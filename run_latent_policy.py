import argparse
import torch
import numpy as np
from pathlib import Path
import sys

base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir / "pacman_agent"))
sys.path.insert(0, str(base_dir / "latent_space"))
sys.path.insert(0, str(base_dir))

from wrappers import make_atari_pacman_env
from dqn_model import build_q_network
from models import Encoder
from latent_action_tracker import LatentStateTracker

def get_action_name(action: int) -> str:
    action_names = {
        0: "NOOP",
        1: "UP",
        2: "RIGHT",
        3: "LEFT",
        4: "DOWN",
        5: "UPRIGHT",
        6: "UPLEFT",
        7: "DOWNRIGHT",
        8: "DOWNLEFT"
    }
    return action_names.get(action, f"UNKNOWN({action})")

def select_next_best_action(q_values: torch.Tensor, visited_actions: set, device: torch.device) -> int:
    q_values_np = q_values.cpu().numpy().flatten()
    sorted_actions = np.argsort(q_values_np)[::-1]
    
    for action in sorted_actions:
        if action not in visited_actions:
            return int(action)
    
    return int(sorted_actions[0])

def prepare_encoder_input(obs: np.ndarray) -> torch.Tensor:
    last_frame = obs[:, :, -1]
    frame_tensor = torch.from_numpy(last_frame).unsqueeze(0).unsqueeze(0).float()
    frame_tensor = frame_tensor / 255.0
    frame_tensor = (frame_tensor - 0.5) / 0.5
    return frame_tensor

def prepare_policy_input(obs: np.ndarray) -> torch.Tensor:
    state_tensor = torch.from_numpy(obs).unsqueeze(0)
    state_tensor = state_tensor.permute(0, 3, 1, 2).contiguous().float()
    return state_tensor

def run_latent_policy(
    policy_path: str = "best_policy.pt",
    encoder_path: str = "latent_space/encoder_model_grayscale.pth",
    num_episodes: int = 5,
    render: bool = True,
    reset_tracker_per_episode: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_path = Path(policy_path)
    encoder_path = Path(encoder_path)
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    
    print(f"Loading policy from {policy_path}")
    policy_state = torch.load(policy_path, map_location=device, weights_only=False)
    policy_net = build_q_network(num_actions=9, frame_stack=4).to(device)
    if isinstance(policy_state, dict) and "model" in policy_state:
        policy_net.load_state_dict(policy_state["model"])
    else:
        policy_net.load_state_dict(policy_state)
    policy_net.eval()
    
    print(f"Loading encoder from {encoder_path}")
    encoder = Encoder(latent_dim=16).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=False))
    encoder.eval()
    
    tracker = LatentStateTracker()
    
    render_mode = "human" if render else None
    env = make_atari_pacman_env(render_mode=render_mode, frame_stack=4, clip_rewards=False)
    
    episode_rewards = []
    total_next_best_actions = 0
    
    try:
        for episode in range(num_episodes):
            if reset_tracker_per_episode:
                tracker.reset()
            
            obs, info = env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            current_lives = info.get("lives", 0)
            episode_next_best_actions = 0
            tracking_started = False
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            while not done and step < 10000:
                frame_number = info.get("frame_number", 0)
                
                if not tracking_started and frame_number > 120:
                    tracking_started = True
                
                state_tensor = prepare_policy_input(obs).to(device)
                encoder_input = prepare_encoder_input(obs).to(device)
                
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    latent_vector = encoder(encoder_input)
                
                if tracking_started:
                    visited_actions = tracker.get_visited_actions(latent_vector)
                else:
                    visited_actions = set()
                
                best_action = q_values.argmax(dim=1).item()
                
                if tracking_started and best_action in visited_actions:
                    action = select_next_best_action(q_values, visited_actions, device)
                    episode_next_best_actions += 1
                    
                    latent_key = tracker.get_latent_key(latent_vector)
                    previous_actions_str = ", ".join([get_action_name(a) for a in sorted(visited_actions)])
                    print(f"\n[LATENT STATE REVISITED]")
                    print(f"Latent vector (from tracker): {np.array(latent_key)}")
                    print(f"Previously taken action(s): {previous_actions_str}")
                    print(f"Policy wanted to take: {get_action_name(best_action)}")
                    print(f"Taking next best action instead: {get_action_name(action)}")
                else:
                    action = best_action
                
                if tracking_started:
                    tracker.record_action(latent_vector, action)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                new_lives = info.get("lives", current_lives)
                if new_lives < current_lives:
                    tracker.reset()
                    tracking_started = False
                current_lives = new_lives
                
                episode_reward += reward
                step += 1
                
                if render and step % 50 == 0:
                    print(f"Step {step}: action={action}, reward={reward}, total_reward={episode_reward:.1f}")
            
            episode_rewards.append(episode_reward)
            total_next_best_actions += episode_next_best_actions
            print(f"Episode {episode + 1} finished: {step} steps, total reward: {episode_reward:.1f}, next_best_actions: {episode_next_best_actions}")
        
        avg_reward = np.mean(episode_rewards)
        print(f"\nEvaluation complete!")
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.1f}")
        print(f"Rewards: {[f'{r:.1f}' for r in episode_rewards]}")
        print(f"Total next best actions triggered: {total_next_best_actions} (avg: {total_next_best_actions/num_episodes:.1f} per episode)")
        
    finally:
        env.close()
    
    return episode_rewards

def main():
    parser = argparse.ArgumentParser(description="Run policy with latent space loop prevention")
    parser.add_argument("--policy", type=str, default="best_policy.pt", help="Path to policy checkpoint")
    parser.add_argument("--encoder", type=str, default="latent_space/encoder_model_grayscale.pth", help="Path to encoder model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--no-reset-tracker", action="store_true", help="Don't reset tracker between episodes")
    
    args = parser.parse_args()
    
    run_latent_policy(
        policy_path=args.policy,
        encoder_path=args.encoder,
        num_episodes=args.episodes,
        render=not args.no_render,
        reset_tracker_per_episode=not args.no_reset_tracker
    )

if __name__ == "__main__":
    main()

