# Author: Carlton Brady

import random
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from prioritized_replay_memory import PrioritizedReplayMemory
from transition_memory import TransitionMemory


class DQNLunarLanderAgent:
    def __init__(self, epsilon, min_epsilon, decay_rate, learning_rate, tau, gamma, batch_size,
                 q_network, target_network, max_memory_length):
        self.experience_memory = deque(maxlen=max_memory_length)
        self.prioritized_memory = PrioritizedReplayMemory(max_length=max_memory_length, alpha=0.6, beta=0.4)
        self.q_network = q_network
        self.target_network = target_network
        # epsilon is the probability of taking a random action
        self.epsilon = epsilon
        # lowest epsilon is allowed to go during training
        self.min_epsilon = min_epsilon
        # rate at which epsilon decays each episode
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        # gamma is the discount factor
        self.gamma = gamma
        # tau is the weighting of the target network parameters when updating them with the
        # regular q network parameters. Tau = 1.0 is no longer double DQN, since it just copies
        # everything as is from the regular q network to the target network.
        self.tau = tau
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        # Huber loss reduces sensitivity to outliers
        # self.criterion = nn.SmoothL1Loss()
        self.loss_history = []
        self.total_actions = 0
        self.total_training_episodes = 0
        self.action_counts = {'0': 0,
                              '1': 0,
                              '2': 0,
                              '3': 0}
        self.action_distribution = {'0': 0.0,
                                    '1': 0.0,
                                    '2': 0.0,
                                    '3': 0.0}

    def decay_epsilon(self):
        # enforce a minimum epsilon during training
        self.epsilon = max(self.epsilon*self.decay_rate, self.min_epsilon)

    def update_action_distribution(self):
        for key in self.action_distribution.keys():
            self.action_distribution[key] = self.action_counts[key]/self.total_actions

    def save_model(self, filename):
        print("Saving Q network...")
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        network_state = {
            'net': self.q_network.state_dict(),
            'target': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'total_training_episodes': self.total_training_episodes
        }
        torch.save(network_state, f'./checkpoints/{filename}.pth')
        print("Save complete!")

    def load_model(self, filename):
        print("Loading model from checkpoint...")
        checkpoint = torch.load(f'./checkpoints/{filename}.pth')  # load checkpoint
        self.q_network.load_state_dict(checkpoint['net'])
        self.target_network.load_state_dict(checkpoint['target'])
        self.epsilon = checkpoint['epsilon']
        self.total_training_episodes = checkpoint['total_training_episodes']
        print("Load complete!")

    def select_action(self, state):
        """
        Using e-greedy exploration.
        Take a random action with probability epsilon,
        otherwise take the action with the highest value given the current state (according to the Q-network)
        :return: Discrete action, int in range 0-4 inclusive
        """
        if random.random() <= self.epsilon:
            action = random.randint(0, 3)
        else:
            # Feed forward the q network and take the action with highest q value
            self.q_network.eval()
            with torch.no_grad():
                qs = self.q_network(torch.tensor(state, dtype=torch.float32))
                action = np.argmax(qs.detach().numpy())
        self.q_network.train()
        self.total_actions += 1
        self.action_counts[str(action)] = self.action_counts[str(action)] + 1
        return action

    def sample_random_experience(self, n):
        """
        Randomly sample n transitions from the experience replay memory into a batch for training
        :param n: number of transition experiences to randomly sample
        :return: tuple of tensor batches of each TransitionMemory attribute
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        experience_sample = random.sample(self.experience_memory, n)
        for memory in experience_sample:
            states.append(memory.state)
            actions.append(memory.action)
            rewards.append([memory.reward])
            next_states.append(memory.next_state)
            dones.append([memory.done])

        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.int8))

    def push_memory(self, memory):
        """Push a transition memory object onto the experience deque"""
        assert (isinstance(memory, TransitionMemory))
        self.experience_memory.append(memory)

    def do_training_update(self):
        if self.batch_size == 0 or len(self.experience_memory) < self.batch_size:
            return
        # Sample experience
        states, actions, rewards, next_states, dones = self.sample_random_experience(n=self.batch_size)
        # Get q values for the current state
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).detach()
        max_next_q = next_q.max(1)[0].unsqueeze(1)
        assert (rewards.size() == max_next_q.size())
        assert (dones.size() == max_next_q.size())
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        target_q = target_q.detach()
        target_q = target_q.squeeze()
        assert (current_q.size() == target_q.size())
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()

    def do_prioritized_training_update(self, frame):
        if self.batch_size == 0 or len(self.prioritized_memory.memory) < self.batch_size:
            return
        # Sample prioritized experience
        states, actions, rewards, next_states, dones, importance_sampling_weights, selected_indices = self.prioritized_memory.sample(self.batch_size)
        # Get q values for the current state
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).detach()
        max_next_q = next_q.max(1)[0].unsqueeze(1)
        assert (rewards.size() == max_next_q.size())
        assert (dones.size() == max_next_q.size())
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        target_q = target_q.detach()
        target_q = target_q.squeeze()
        assert (current_q.size() == target_q.size())
        loss = (current_q - target_q).pow(2)
        assert (loss.size() == importance_sampling_weights.size())
        # Multiply the TD errors by the importance sampling weights
        loss = loss * importance_sampling_weights
        new_priorities = loss + 0.00001
        loss = torch.mean(loss)
        self.optimizer.zero_grad()
        self.loss_history.append(loss.item())
        loss.backward()
        self.optimizer.step()
        # Update priorities for the indices selected in the batch
        self.prioritized_memory.update_priorities(selected_indices, new_priorities.detach().numpy())
        self.prioritized_memory.anneal_beta(frame)

    def update_target_network(self):
        for source_parameters, target_parameters in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_parameters.data.copy_(self.tau * source_parameters.data + (1.0 - self.tau) * target_parameters.data)


class DQN(nn.Module):
    def __init__(self, arch="BASELINE"):
        super(DQN, self).__init__()
        self.arch = arch
        if self.arch == "WIDE":
            # Wider
            self.fc1 = nn.Linear(8, 64)
            self.fc2 = nn.Linear(64, 128)
            self.fc3 = nn.Linear(128, 4)
        elif self.arch == "DEEP":
            # Deeper
            self.fc1 = nn.Linear(8, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 128)
            self.fc4 = nn.Linear(128, 4)
        else:
            # Baseline
            self.fc1 = nn.Linear(8, 32)
            self.fc2 = nn.Linear(32, 64)
            self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        if self.arch == "DEEP":
            # deeper
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        else:
            # baseline
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x
