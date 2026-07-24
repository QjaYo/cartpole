import gymnasium as gym
import collections
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MODEL_PATH = Path(__file__).with_name("cartpole_dqn_10000.pt")

# hyperparameters
learning_rate   = 0.0005
gamma           = 0.98
buffer_limit    = 50000
batch_size      = 32
episode_limit   = 10000

# classes
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states, actions, rewards, next_states, done_masks = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            states.append(s)
            actions.append([a])
            rewards.append([r])
            next_states.append(s_prime)
            done_masks.append([done_mask])

        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        done_masks = torch.tensor(done_masks, dtype=torch.float32)
        
        return states, actions, rewards, next_states, done_masks
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super().__init__() # nn.Module.init(self)
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon): # choose one. move left or move right; obs: current state s; return: [Q(s,0), Q(s,1)]
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item() # argmax: pytorch tensor; .item: convert tensor to python number

def train(q, q_target, buffer, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = buffer.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a) # gather: 0: select row; 1: select column; 과거 기록의 상태 s에서 실제로 했던 행동 a의 가치를 현재 네트워크에서 예측한 값
        with torch.no_grad():
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # max: (0): find max value of each row; (1): find max value of each column; [0]: return max value; [1]: return index of max value; unsqueeze: add dimension (0): add front, (1): add back
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward() # compute gradient of loss with respect to parameters
        optimizer.step() # update parameters using gradient descent
    
def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    buffer = ReplayBuffer()

    target_update_interval = 20
    score = 0.0 # +=1 per step
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(episode_limit):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) # n_epi<1400: linear annealing, n_epi>1400: 0.01 fixed
        s, info = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if terminated else 1.0
            buffer.put((s, a, r/100.0, s_prime, done_mask))

            s = s_prime
            score += r

            if done:
                break

        if buffer.size() > 2000: # after sufficient experience
            train(q, q_target, buffer, optimizer)
        
        if (n_epi + 1) % target_update_interval == 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode: {}, score: {:.1f}, n_buffer: {}, epsilon: {:.4f}".format(n_epi, score/target_update_interval, buffer.size(), epsilon))
            score = 0.0

    torch.save(q.state_dict(), MODEL_PATH)
    env.close()

if __name__ == '__main__':
    main()
