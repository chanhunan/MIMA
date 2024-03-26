import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from pygcn.pygcn.train import batch_size


class LinearEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_tensor):
        encoded_tensor = self.linear(input_tensor)
        return encoded_tensor


class Math_Path_Layer(nn.Module):
    def __init__(self, in_dim, heads, self_attention):
        super(self_attention, self).__init__()
        super().__init__()
        self.in_dim = in_dim
        self.heads = heads
        self.linear = nn.Linear(in_dim, in_dim * heads)


    def feature_function(G, meta_path):

        path_nodes = []
        current_node = meta_path[0]
        path_nodes.append(current_node)

        for i in range(1, len(meta_path)):
            next_node = meta_path[i]
            if G.has_edge(current_node, next_node):
                path_nodes.append(next_node)
                current_node = next_node
            else:
                break
        return path_nodes


    def forward(self, path_nodes, seq_length):

        in_feature = self.linear(path_nodes).view(-1, self.heads, self.in_dim)
        in_feature = torch.softmax(in_feature, dim=-1)
        in_feature = in_feature.mean(dim=1)

        input_size = len(path_nodes)
        output_size = 256
        linear_encoder = LinearEncoder(input_size, output_size)

        mean_tensor = torch.randn(batch_size, seq_length, input_size)

        linear_encoded_tensor = linear_encoder(mean_tensor)

        weight_matrix = torch.randn(output_size, input_size)

        edge_feature = torch.matmul(linear_encoded_tensor, weight_matrix)
        mid_feature = torch.cat([edge_feature.mean(dim=0), edge_feature.max(dim=0).values], dim=-1)

        feature = torch.cat([in_feature, mid_feature], dim=1)

        return feature