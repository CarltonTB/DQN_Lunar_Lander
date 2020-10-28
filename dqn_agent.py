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
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        """
        Using e-greedy exploration.
        Take a random action with probability epsilon,
        otherwise take the action with the highest value given the current state (according to the Q-network)
        :return: Discrete action, int in range 0-4 inclusive
        """
        if random.random() <= self.epsilon:
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
            next_states.append(next_states)
            dones.append(memory.done)

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                dones)

    def push_memory(self, memory):
        """Push a transition memory object onto the experience deque"""
        assert (isinstance(memory, TransitionMemory))
        self.experience_memory.append(memory)


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
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
