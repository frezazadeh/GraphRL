import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv, LayerNorm, global_add_pool
import numpy as np
import random

class GraphReinforceAgent(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_layer_dimension=256, learning_rate=0.0005):
        super(GraphReinforceAgent, self).__init__()
        self.graph_convolution_layer = GCNConv(2, hidden_layer_dimension)
        self.hidden_linear_layer = nn.Linear(hidden_layer_dimension, hidden_layer_dimension)
        self.output_layer = nn.Linear(hidden_layer_dimension, output_dimension)
        self.normalization_layer = LayerNorm(hidden_layer_dimension)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.experience_memory = []

    def store_transition(self, transition):
        self.experience_memory.append(transition)

    def forward(self, node_features, edge_index):
        node_features = F.relu(self.graph_convolution_layer(node_features, edge_index))
        node_features = global_add_pool(self.normalization_layer(node_features), torch.LongTensor([0] * 4).to(node_features.device))
        node_features = F.relu(self.hidden_linear_layer(node_features))
        output = self.output_layer(node_features)
        return F.log_softmax(output, dim=1)

    def optimize(self, discount_factor):
        cumulative_reward, discounted_rewards, running_discounted_reward = 0, [], 0
        for reward, log_prob in reversed(self.experience_memory):
            running_discounted_reward = reward + discount_factor * running_discounted_reward
            discounted_rewards.append(running_discounted_reward)
        discounted_rewards = np.array(discounted_rewards)
        rewards_mean, rewards_std_dev = discounted_rewards.mean(), discounted_rewards.std()
        self.optimizer.zero_grad()

        for reward, log_prob in reversed(self.experience_memory):
            cumulative_reward = reward + discount_factor * cumulative_reward
            policy_loss = -log_prob * ((cumulative_reward - rewards_mean) / rewards_std_dev)
            policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.experience_memory = []

class Utils:
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def save_model(model, path='model_checkpoint.pth'):
        torch.save(model.state_dict(), path)
