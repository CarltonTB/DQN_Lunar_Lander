# Author: Carlton Brady

import gym
import time
from dqn_agent import *
import torch


def run_lunar_lander(agent):
    """Run the lunar lander v2 environment"""
    start = time.time()
    # Make lunar lander v2 environment
    env = gym.make('LunarLander-v2')
    env.reset()
    # print(env.action_space)
    # print(env.observation_space)
    for episode in range(100):
        state = None
        for t in range(1000):
            if t > 0:
                state = observation
            # env.render()
            # take a random action
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            next_state = observation
            # TODO: what should the state be if it's the first timestep? all zeros?
            agent.push_memory(TransitionMemory(state, action, reward, next_state, done))
            if done:
                # print(f'Episode {episode} ended in {t + 1} timesteps')
                break
        env.reset()
    env.close()
    print(f'Time elapsed: {time.time() - start} seconds')
    print(len(agent.experience_memory))


if __name__ == "__main__":
    agent = DQNLunarLanderAgent(epsilon=0.1, q_network=DQN(), max_memory_length=10000)
    run_lunar_lander(agent)
