# Author: Carlton Brady

import gym
import time
from dqn_agent import *
import torch
import numpy as np


def run_lunar_lander(agent):
    """Run the lunar lander v2 environment"""
    start = time.time()
    # Make lunar lander v2 environment
    env = gym.make('LunarLander-v2')
    env.reset()
    for episode in range(200):
        state = None
        for t in range(10000):
            if t > 0:
                state = observation
            if episode > 150:
                env.render()
            # take a random action
            action = agent.select_action(state)
            print(action)
            observation, reward, done, info = env.step(action)
            next_state = observation
            agent.push_memory(TransitionMemory(state, action, reward, next_state, done)) if state is not None else print("No state")
            # train the network:
            if t > 50:
                agent.do_training_update(batch_size=50)
            if done:
                # print(f'Episode {episode} ended in {t + 1} timesteps')
                break
        agent.decay_epsilon(decay_rate=0.99)
        env.reset()
    env.close()
    print(f'Time elapsed: {time.time() - start} seconds')
    print(len(agent.experience_memory))


if __name__ == "__main__":
    ll_agent = DQNLunarLanderAgent(epsilon=1.0, learning_rate=0.00001, gamma=0.99, q_network=DQN(), max_memory_length=100000)
    run_lunar_lander(ll_agent)
