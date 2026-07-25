import argparse
from pathlib import Path

import gymnasium as gym
import pygame
import torch

from main import MODEL_DIR, Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained CartPole policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "cartpole_policy_10000.pt",
        help="Path to a saved policy state_dict.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    policy = Policy()
    policy.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    policy.eval()

    env = gym.make("CartPole-v1", render_mode="human")
    running = True

    for _ in range(args.episodes):
        if not running:
            break

        state, info = env.reset()
        done = False

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = policy(state_tensor).argmax().item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
