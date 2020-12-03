# Author: Carlton Brady

import random


class RandomAgent:

    def __init__(self):
        self.epsilon = None

    def select_action(self, state):
        return random.randint(0, 3)
