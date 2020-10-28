# Author: Carlton Brady

import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DQNLunarLanderAgent:
    def __init__(self, epsilon, learning_rate, gamma, q_network, max_memory_length):
        self.experience_memory = deque(maxlen=max_memory_length)
        self.q_network = q_network
        # epsilon is the probability of taking a random action
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        # gamma is the discount factor
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.loss_history = []

    def decay_epsilon(self, decay_rate):
        self.epsilon = self.epsilon*decay_rate

    def select_action(self, state):
        """
        Using e-greedy exploration.
        Take a random action with probability epsilon,
        otherwise take the action with the highest value given the current state (according to the Q-network)
        :return: Discrete action, int in range 0-4 inclusive
        """
        if state is None:
            return 0
        elif random.random() <= self.epsilon:
            return random.randint(0, 3)
        else:
            # Feed forward the q network and take the action with highest q value
            qs = self.q_network(torch.tensor(state, dtype=torch.float32))
            return np.argmax(qs.detach().numpy())

    def sample_random_experience(self, n):
        """
        Randomly sample n transitions from the experience replay memory into a batch for training
        :param n: number of transition experiences to randomly sample
        :return: tuple of tensor batches of each TransitionMemory attribute
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        experience_sample = random.sample(self.experience_memory, n)
        for memory in experience_sample:
            states.append(memory.state)
            actions.append(memory.action)
            rewards.append(memory.reward)
            next_states.append(memory.next_state)
            dones.append(memory.done)

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def push_memory(self, memory):
        """Push a transition memory object onto the experience deque"""
        assert (isinstance(memory, TransitionMemory))
        self.experience_memory.append(memory)

    def do_training_update(self, batch_size):
        if batch_size == 0:
            return
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = self.sample_random_experience(n=batch_size)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.q_network(next_states)
        max_q = torch.max(next_q).item()
        expected_q = rewards + (1 - dones) * self.gamma * max_q
        expected_q = expected_q.detach()
        assert (current_q.size() == expected_q.size())
        loss = self.criterion(current_q, expected_q)
        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()


class TransitionMemory:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
