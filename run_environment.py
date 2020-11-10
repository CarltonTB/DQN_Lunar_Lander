# Author: Carlton Brady

import gym
import time
from dqn_agent import *
import torch
import matplotlib.pyplot as plt


def train_lunar_lander(agent, episodes):
    """Run the lunar lander v2 environment"""
    start = time.time()
    # Make lunar lander v2 environment
    env = gym.make('LunarLander-v2')
    env.reset()
    env.seed(0)
    score_history = []
    last_hundred = deque(maxlen=100)
    for episode in range(episodes):
        score = 0
        state = env.reset()
        for t in range(1000):
            # env.render()
            # take an action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_memory(TransitionMemory(state, action, reward, next_state, done))
            # Update the q network and target network parameters
            agent.do_training_update()
            score += reward
            state = next_state
            if done:
                # print(f'Episode {episode} ended in {t + 1} timesteps')
                break
        agent.decay_epsilon()
        env.reset()
        score_history.append(score)
        last_hundred.append(score)
        score_sum = 0
        for score in last_hundred:
            score_sum += score
        mean_last_hundred = score_sum/100
        if episode % 100 == 0:
            agent.update_action_distribution()
            print(f'Episode {episode} complete!')
            print("Action Distribution:")
            print(agent.action_distribution)
            print(f'Epsilon: {agent.epsilon}')
            print(f"Avg. score last 100: {mean_last_hundred}")
            agent.save_model()
            print("\n")
        if mean_last_hundred > 200.0:
            agent.save_model()
            print("SOLVED!")
            break
    env.close()
    print(f'Time elapsed: {time.time() - start} seconds')
    plot_score(score_history)


def test_lunar_lander(agent, episodes, load_from_checkpoint=False, render=False):
    if load_from_checkpoint:
        checkpoint = torch.load('./checkpoint/ckpt.pth')  # load checkpoint
        agent.q_network.load_state_dict(checkpoint['net'])
    env = gym.make('LunarLander-v2')
    env.reset()
    env.seed(0)
    score_history = []
    last_hundred = deque(maxlen=100)
    mean_last_hundred = None
    agent.epsilon = 0
    for episode in range(episodes):
        score = 0
        state = env.reset()
        for t in range(1000):
            if render:
                env.render()
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                # print(f'Episode {episode} ended in {t + 1} timesteps')
                break
        print(f'Episode score: {score}')
        score_history.append(score)
        last_hundred.append(score)
        score_sum = 0
        for score in last_hundred:
            score_sum += score
        mean_last_hundred = score_sum/100
        env.reset()
    env.close()
    print(f'Mean score last 100 episodes: {mean_last_hundred}')
    plot_score(score_history)


def plot_score(score):
    plt.figure(figsize=(10, 5))
    plt.title("Score vs. Episodes")
    plt.plot(score, label=score)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()


if __name__ == "__main__":
    ll_agent = DQNLunarLanderAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.995,
                                   learning_rate=0.0001, gamma=0.99, batch_size=64,
                                   tau=0.001, q_network=DQN(), target_network=DQN(), max_memory_length=500000)
    # train_lunar_lander(ll_agent, episodes=1000)
    test_lunar_lander(ll_agent, 100, load_from_checkpoint=True, render=True)
