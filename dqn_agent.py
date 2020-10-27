# Author: Carlton Brady

import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F


class DQNLunarLanderAgent:
    def __init__(self, epsilon, q_network, max_memory_length):
        self.experience_memory = deque(maxlen=max_memory_length)
        self.q_network = q_network
        self.epsilon = epsilon

    def select_action(self):
        """
        Using e-greedy exploration.
        Take a random action with probability epsilon,
        otherwise take the action with the highest value given the current state (according to the Q-network)
        :return: Discrete action, int in range 0-4 inclusive
        """
        if random.random() <= self.epsilon:
            return random.randint(0, 4)
        else:
            # Feed forward the q network and take the action with highest q value
            return 0

    def sample_random_experience(self, n):
        """
        Randomly sample n transitions from the experience replay memory
        :param n: number of transition experiences to randomly sample
        :return: list of n transition experiences
        """
        return random.sample(self.experience_memory, n)

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
