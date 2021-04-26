# Author: Carlton Brady

import gym
import time
from datetime import datetime
from dqn_agent import *
from random_agent import *
import matplotlib.pyplot as plt
import sys


def train_lunar_lander(agent, episodes, save_filename, load_from_checkpoint=True):
    """Run the lunar lander v2 environment"""
    if load_from_checkpoint:
        agent.load_model(save_filename)
    start = time.time()
    # Make lunar lander v2 environment
    env = gym.make('LunarLander-v2')
    env.reset()
    env.seed(0)
    score_history = []
    last_hundred = deque(maxlen=100)
    rolling_means = []
    for episode in range(agent.total_training_episodes, agent.total_training_episodes + episodes):
        score = 0
        state = env.reset()
        for t in range(1000):
            # env.render()
            # take an action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push_memory(TransitionMemory(state, action, reward, next_state, done))
            # Update the q network parameters
            agent.do_training_update()
            # Update target network parameters every 5 timesteps
            if t % 5 == 0:
                agent.update_target_network()
            score += reward
            state = next_state
            # Try decaying faster
            agent.decay_epsilon()
            if done:
                # print(f'Episode {episode} ended in {t + 1} timesteps')
                break
        agent.total_training_episodes += 1
        env.reset()
        score_history.append(score)
        last_hundred.append(score)
        score_sum = 0
        for score in last_hundred:
            score_sum += score
        mean_last_hundred = score_sum/100
        rolling_means.append(mean_last_hundred)
        if episode % 100 == 0:
            agent.update_action_distribution()
            print(f'Training episode {episode} complete!')
            print("Action Distribution:")
            print(agent.action_distribution)
            print(f'Epsilon: {agent.epsilon}')
            print(f"Avg. score last 100: {mean_last_hundred}")
            agent.save_model(save_filename)
            print("\n")
        # if mean_last_hundred > 200.0:
        #     print("SOLVED!")
        #     print(f'Total training episodes: {agent.total_training_episodes}')
        #     agent.save_model(save_filename)
        #     break
    env.close()
    print(f'Time elapsed: {time.time() - start} seconds')
    plot_score(score_history, rolling_means, f"training_plot_{save_filename}")


def test_lunar_lander(agent, episodes, filename, load_from_checkpoint=False, render=False):
    if load_from_checkpoint:
        agent.load_model(filename)
    env = gym.make('LunarLander-v2')
    env.reset()
    env.seed(0)
    score_history = []
    last_hundred = deque(maxlen=100)
    mean_last_hundred = None
    rolling_means = []
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
        mean_last_hundred = score_sum/len(last_hundred)
        rolling_means.append(mean_last_hundred)
        env.reset()
    env.close()
    print(f'Mean score last 100 episodes: {mean_last_hundred}')
    plot_score(score_history, rolling_means, f"testing_plot_{filename}")


def plot_score(score, rolling_mean, filename):
    plt.figure(figsize=(10, 5))
    plt.title("Score vs. Episodes")
    plt.plot(score, label="Score")
    plt.plot(rolling_mean, label="Mean last 100 episodes")
    plt.ylabel("Score")
    plt.xlabel("Episode")
    now = datetime.now()
    datetime_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    plt.savefig(f"./graphs/{filename}")
    plt.show()


if __name__ == "__main__":
    ll_agent = DQNLunarLanderAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.995,
                                   learning_rate=0.0001, gamma=0.99, batch_size=64,
                                   tau=0.001, q_network=DQN(), target_network=DQN(), max_memory_length=500000)
    ll_agent_wide = DQNLunarLanderAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.995,
                                        learning_rate=0.0001, gamma=0.99, batch_size=64,
                                        tau=0.001, q_network=DQN(arch="WIDE"), target_network=DQN(arch="WIDE"),
                                        max_memory_length=500000)
    # train_lunar_lander(ll_agent, episodes=500, save_filename="ddqn8-32-64-4-half-trained", load_from_checkpoint=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-4-fast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-32-64-4-slower-decay", load_from_checkpoint=True, render=True)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-4-superfast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-32-64-4-fast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-32-64-4-superfast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-64-128-4-superfast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-64-128-4-slower-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-64-128-4-fast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-64-128-4-superfast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-64-128-4-fast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-64-128-4-slow-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-32-64-128-4-superfast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="dqn8-32-64-128-4-fast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-128-4-superfast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-128-4-fast-decay", load_from_checkpoint=True, render=False)
    # test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-128-4-slow-decay", load_from_checkpoint=True, render=True)

    demo_choice = input("Enter number corresponding to desired demo:\n"
                        "1. Random Agent\n"
                        "2. Partially-Trained\n"
                        "3. Trained enough to solve\n"
                        "4. 3000 Episodes of training\n")
    demo_choice = int(demo_choice)
    # DEMO RUNS
    if demo_choice == 1:
        # Random Agent
        test_lunar_lander(RandomAgent(), 100, filename="test_rand_agent", load_from_checkpoint=False, render=True)
    elif demo_choice == 2:
        # Half Trained
        # 400 epsiodes of training, hasn't yet explored enough to know it should land on the pad, just knows to avoid crashing
        test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-4-half-trained", load_from_checkpoint=True, render=True)
    elif demo_choice == 3:
        # Trained just enough to solve
        test_lunar_lander(ll_agent, 100, filename="ddqn8-32-64-4-fast-decay", load_from_checkpoint=True, render=True)
    elif demo_choice == 4:
        # Long training (3000 episodes)
        test_lunar_lander(ll_agent_wide, 100, filename="ddqn8-32-64-128-4-slow-decay-long-training", load_from_checkpoint=True, render=True)
    else:
        print("Invalid choice")















