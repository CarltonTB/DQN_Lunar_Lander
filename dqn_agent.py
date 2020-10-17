# Author: Carlton Brady

from random import *


class DQNLunarLanderAgent:

    def __init__(self, epsilon):
        self.experience = []
        self.q_network = None
        self.epsilon = 0.1

    def select_action(self):
        # e-greedy exploration
        if random() <= self.epsilon:
            # Take a random action with probability epsilon
            return randint(0, 4)
        # Take the action with the highest value given the current state
        return 0
