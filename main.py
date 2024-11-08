import gym
import torch
from torch.distributions import Categorical
import numpy as np

from graphmaker import GraphDataProcessor
from model import GraphReinforceAgent, Utils

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    seed = 10
    Utils.set_seed(seed)
    env.seed(seed)

    # Hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    learning_rate = 0.0005
    episodes = 200
    gamma = 0.99
    print_interval = 10

    agent = GraphReinforceAgent(state_dim, action_dim, hidden_layer_dimension=128, learning_rate=learning_rate).to(device)
    score_list = []
    index_pairs = GraphDataProcessor.construct_index_pairs(state_dim)

    for episode in range(episodes):
        state = env.reset()
        score = 0
        done = False

        while not done:
            graph_data = GraphDataProcessor.create_graph_data(state, index_pairs)
            action_probs = agent(graph_data.x.to(device), graph_data.edge_index.to(device))
            action_distribution = Categorical(torch.exp(action_probs))
            action = action_distribution.sample()

            next_state, reward, done, _ = env.step(action.item())
            agent.store_transition((reward, action_probs[0][action]))
            state = next_state
            score += reward

        agent.optimize(gamma)
        score_list.append(score)

        if episode % print_interval == 0 and episode != 0:
            avg_score = sum(score_list[-print_interval:]) / print_interval
            print(f"Episode {episode}, Avg Score: {avg_score}")

    env.close()
