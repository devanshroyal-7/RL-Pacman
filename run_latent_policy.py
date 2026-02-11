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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
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

def select_next_best_action(model: PPO, obs: np.ndarray, visited_actions: set) -> int:
    """
    Select the next best action based on PPO action probabilities,
    excluding already visited actions.
    """
    # Get action distribution from PPO policy
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        action_probs = distribution.distribution.probs.cpu().numpy().flatten()
    
    # Sort actions by probability (descending)
    sorted_actions = np.argsort(action_probs)[::-1]
    
    # Find first action not in visited_actions
    for action in sorted_actions:
        if action not in visited_actions:
            return int(action)
    
    # Fallback: return most probable action
    return int(sorted_actions[0])

def prepare_encoder_input(obs: np.ndarray) -> torch.Tensor:
    last_frame = obs[:, :, -1]
    frame_tensor = torch.from_numpy(last_frame).unsqueeze(0).unsqueeze(0).float()
    frame_tensor = frame_tensor / 255.0
    frame_tensor = (frame_tensor - 0.5) / 0.5
    return frame_tensor

def prepare_policy_input(obs: np.ndarray) -> np.ndarray:
    """
    Prepare observation for PPO. PPO expects observations in CHW format.
    """
    # obs is already in HWC format from the environment
    # Convert to CHW format (channels first)
    if len(obs.shape) == 3:
        # Single frame: HWC -> CHW
        obs = np.transpose(obs, (2, 0, 1))
    elif len(obs.shape) == 4:
        # Batch: NHWC -> NCHW
        obs = np.transpose(obs, (0, 3, 1, 2))
    return obs

def run_latent_policy(
    policy_path: str = "ppo_pacman.zip",
    encoder_path: str = "latent_space/encoder_model_grayscale.pth",
    num_episodes: int = 5,
    render: bool = True,
    reset_tracker_per_episode: bool = True,
    device: str = "cuda:0",
    next_action: bool = False
):
    device_torch = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    print(f"Using device: {device_torch}")
    
    policy_path = Path(policy_path)
    encoder_path = Path(encoder_path)
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    
    print(f"Loading policy from {policy_path}")
    def _make_env():
        env = make_atari_pacman_env(render_mode=None, frame_stack=4)
        return env
    
    dummy_vec_env = DummyVecEnv([_make_env])
    vec_transpose_img = VecTransposeImage(dummy_vec_env)

    policy = PPO.load(str(policy_path), env=vec_transpose_img, device=device)
    policy.set_env(vec_transpose_img)
    
    print(f"Loading encoder from {encoder_path}")
    encoder = Encoder(latent_dim=16).to(device_torch)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device_torch, weights_only=False))
    encoder.eval()
    
    tracker = LatentStateTracker()
    
    render_mode = "human" if render else None
    env = make_atari_pacman_env(render_mode=render_mode, frame_stack=4)
    
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
            revisiting_count = 0
            death_frame = None  # Track when death occurred
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            while not done and step < 10000:
                frame_number = info.get("frame_number", 0)
                # print("Frame Number: ", frame_number)
                
                if not tracking_started and frame_number > 252:
                    # If there was a death, wait 80 frames before restarting tracking
                    if death_frame is None or (frame_number - death_frame) > 150:
                        tracking_started = True
                
                # Prepare observation for PPO (CHW format)
                obs_for_ppo = prepare_policy_input(obs)
                encoder_input = prepare_encoder_input(obs).to(device_torch)
                
                with torch.no_grad():
                    latent_vector = encoder(encoder_input)
                
                if tracking_started:

                    visited_actions = tracker.get_visited_actions(latent_vector)
                else:
                    visited_actions = set()
                
                # Get action from PPO
                best_action, _ = policy.predict(obs_for_ppo, deterministic=False)
                best_action = int(best_action)
                
                if tracking_started and tracker.has_visited(latent_vector):
                    if next_action:
                        action = select_next_best_action(policy, obs_for_ppo, visited_actions)
                    else:
                        action = best_action
                    episode_next_best_actions += 1
                    revisiting_count += 1
                    latent_key = tracker.get_latent_key(latent_vector)
                    previous_actions_str = ", ".join([get_action_name(a) for a in sorted(visited_actions)])
                    print(f"\n[LATENT STATE REVISITED]")
                    print(f"revisiting count: {revisiting_count}")
                    # print(f"Latent vector (from tracker): {np.array(latent_key)}")
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
                    death_frame = frame_number  # Record the frame when death occurred
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
        vec_transpose_img.close()
    
    return episode_rewards

def main():
    parser = argparse.ArgumentParser(description="Run PPO policy with latent space loop prevention")
    parser.add_argument("--policy", type=str, default="ppo_pacman.zip", help="Path to PPO checkpoint (.zip)")
    parser.add_argument("--encoder", type=str, default="latent_space/encoder_model_grayscale.pth", help="Path to encoder model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--no-reset-tracker", action="store_true", help="Don't reset tracker between episodes")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
    parser.add_argument("--next-action", action="store_true", help="Use next best action instead of best action")
    
    args = parser.parse_args()
    
    run_latent_policy(
        policy_path=args.policy,
        encoder_path=args.encoder,
        num_episodes=args.episodes,
        render=not args.no_render,
        reset_tracker_per_episode=not args.no_reset_tracker,
        device=args.device,
        next_action=args.next_action
    )

if __name__ == "__main__":
    main()

