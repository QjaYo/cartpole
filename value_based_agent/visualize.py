import gymnasium as gym
import pygame
import torch

from main import MODEL_PATH, Qnet

q = Qnet()
q.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)
q.eval()

env = gym.make("CartPole-v1", render_mode="human")

running = True

for episode in range(5):
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
            action = q(state_tensor).argmax().item()

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()
