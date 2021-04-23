# Naive Implementation of prioritized experience replay
# Could be implemented more efficiently with a SumTree data structure

from collections import deque
import numpy as np
from dqn_agent import TransitionMemory
import torch


class PrioritizedReplayMemory:

    def __init__(self, max_length, alpha, beta):
        self.memory = deque(max_length = max_length)
        self.max_length = max_length
        # hyper param used to determine how much TD error determines prioritization of a transition experience
        self.alpha = alpha
        # hyper param for importance sampling that is decayed over time throughout training
        self.beta = beta
        # stores the temporal difference errors for the corresponding transition experience at each index
        self.priorities = np.zeros(max_length, dtype=np.float32)

    def push(self, transition_memory, priority):
        """
        :param transition_memory: TransitionMemory object from a single agent interaction with the environment
        :param priority: TD error of the provided transition_memory
        :return: None
        """
        assert (isinstance(transition_memory, TransitionMemory))
        self.memory.append(transition_memory)
        self.priorities.append(priority)

    def sample(self, n):
        """
        :param n: number of
        :return: tuple of tensors containing states, actions, rewards, next states, and dones
        """
        if self.max_length < self.priorities.size:
            # Truncate the priorities array so it matches the size of the memory array
            self.priorities = self.priorities[self.priorities.size-self.max_length:]
            assert(len(self.memory) == self.priorities.size)

        selection_probabilities = self.priorities ** self.alpha
        selection_probabilities = selection_probabilities / selection_probabilities.sum()

        select_indices = np.random.choice(len(self.memory), n, p=selection_probabilities)
        sampled = [self.memory[i] for i in select_indices]
        # TODO: compute the importance sampling weights and return everything as tensors

