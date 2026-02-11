from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from wrappers import make_atari_pacman_env


def make_vec_env(frame_stack: int, seed: int, render_mode: str | None):
    def _make_env():
        env = make_atari_pacman_env(render_mode=render_mode, frame_stack=frame_stack, episodic_life=False)
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([_make_env])
    env = VecTransposeImage(env)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on Ms. Pacman.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.zip).")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g., cuda:0.")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of stacked frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    render_mode = "human" if args.render else None
    env = make_vec_env(frame_stack=args.frame_stack, seed=args.seed, render_mode=render_mode)

    model = PPO.load(ckpt_path, device=args.device)

    returns: list[float] = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])
            done = bool(dones[0])

        returns.append(ep_reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}")

    avg = sum(returns) / len(returns)
    print(f"Average reward over {args.episodes} episodes: {avg:.2f}")


if __name__ == "__main__":
    main()

