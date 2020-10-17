# Author: Carlton Brady

import gym
import time


def run_lunar_lander():
    """Run the lunar lander v2 environment"""
    start = time.time()
    # Make lunar lander v2 environment
    env = gym.make('LunarLander-v2')
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    for episode in range(100):
        for t in range(1000):
            # env.render()
            # take a random action
            randAction = env.action_space.sample()
            observation, reward, done, info = env.step(randAction)
            if done:
                # print(f'Episode {episode} ended in {t + 1} timesteps')
                break

        env.reset()

    env.close()
    print(f'Time elapsed: {time.time() - start} seconds')


if __name__ == "__main__":
    run_lunar_lander()
