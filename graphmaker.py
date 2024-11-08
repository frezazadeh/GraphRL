#Please check https://github.com/keep9oing/GNN_RL for more information.

import torch
from torch_geometric.data import Data
from itertools import permutations

class GraphDataProcessor:
    @staticmethod
    def create_graph_data(data, index_pairs):
        edge_index = list(permutations(range(len(data)), 2))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.tensor([[data[i], data[j]] for i, j in index_pairs], dtype=torch.float)
        return Data(x=node_features, edge_index=edge_index)

    @staticmethod
    def construct_index_pairs(num_nodes):
        index_pairs = [(i, i + 1) for i in range(0, num_nodes - 1, 2)]
        index_pairs.extend((i, num_nodes - i - 1) for i in range(num_nodes // 2))
        return index_pairs
