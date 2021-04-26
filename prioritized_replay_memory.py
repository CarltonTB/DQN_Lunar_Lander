# Naive Implementation of prioritized experience replay
# Could be implemented more efficiently with a SumTree data structure

from collections import deque
import numpy as np
from dqn_agent import TransitionMemory
import torch


class PrioritizedReplayMemory:

    def __init__(self, max_length, alpha, beta):
        self.memory = deque(max_length=max_length)
        self.max_length = max_length
        # hyper param used to determine how much TD error determines prioritization of a transition experience
        self.alpha = alpha
        # hyper param for importance sampling that is decayed over time throughout training
        self.beta = beta
        # stores the temporal difference errors for the corresponding transition experience at each index
        self.priorities = np.zeros(max_length, dtype=np.float32)

    def push(self, transition_memory):
        """
        :param transition_memory: TransitionMemory object from a single agent interaction with the environment
        :param priority: TD error of the provided transition_memory
        :return: None
        """
        assert (isinstance(transition_memory, TransitionMemory))
        self.memory.append(transition_memory)
        # Priority starts out as max priority and is updated in the training function
        if len(self.memory > 0):
            self.priorities.append(self.priorities.max())
        else:
            self.priorities.append(1.0)

    def sample(self, n):
        """
        :param n: number of
        :return: tuple of tensors containing states, actions, rewards, next states, and dones
        """
        if self.max_length < self.priorities.size:
            # Truncate the priorities array so it matches the size of the memory array
            self.priorities = self.priorities[self.priorities.size-self.max_length:]
            assert(len(self.memory) == self.priorities.size)

        selection_probabilities = self.priorities**self.alpha
        selection_probabilities = selection_probabilities / selection_probabilities.sum()

        selected_indices = np.random.choice(len(self.memory), n, p=selection_probabilities)
        sampled = [self.memory[i] for i in selected_indices]
        total_experiences = len(self.memory)
        importance_sampling_weights = ((1/total_experiences) * (1/selection_probabilities[selected_indices]))**self.beta
        # Normalize the weights to range (0, 1) inclusive
        importance_sampling_weights = importance_sampling_weights / np.max(importance_sampling_weights)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition_memory in sampled:
            states.append(transition_memory.state)
            actions.append(transition_memory.action)
            rewards.append([transition_memory.reward])
            next_states.append(transition_memory.next_state)
            dones.append([transition_memory.done])

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.int8),
                torch.tensor(importance_sampling_weights, dtype=torch.float32),
                selected_indices)



