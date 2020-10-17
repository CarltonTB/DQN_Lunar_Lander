# Author: Carlton Brady

import random


class DQNLunarLanderAgent:

    def __init__(self, epsilon):
        self.experience_memory = []
        self.q_network = None
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
            return 0

    def sample_random_experience(self, n):
        """
        Randomly sample n transitions from the experience replay memory
        :param n: number of transition experiences to randomly sample
        :return: list of n transition experiences
        """
        return random.sample(self.experience_memory, n)


class TransitionMemory:

    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
