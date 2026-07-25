import gymnasium as gym
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Hyperparameters
learning_rate   = 0.0002
gamma           = 0.98
episode_limit   = 10000
checkpoint_interval = 2500

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R # return of this time step
            loss = -torch.log(prob) * R # loss of this time step
            loss.backward() # compute gradient of loss with respect to parameters and accumulate the gradients of each parameter
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(episode_limit):
        s, info = env.reset()
        done = False

        while not done:
            prob = pi(torch.from_numpy(s).float()) # e.g. tensor([0.3, 0.7])
            prob_dist = Categorical(prob) # build a categorical distribution based on the probabilities
            a = prob_dist.sample()  # sample an action from the distribution
            s_prime, r, terminated, truncated, info = env.step(a.item())
            done = terminated or truncated
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net()
        if (n_epi+1) % print_interval == 0:
            print("# of episode: {}, avg score: {}".format(n_epi, score/print_interval))
            score = 0.0

        if (n_epi + 1) % checkpoint_interval == 0:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_DIR / f"cartpole_policy_{n_epi + 1}.pt"
            torch.save(pi.state_dict(), model_path)
            print(f"saved model: {model_path}")

    env.close()

if __name__ == "__main__":
    main()
