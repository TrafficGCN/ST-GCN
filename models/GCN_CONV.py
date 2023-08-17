# Author 2023 Thomas Fink

import torch.nn as nn
import torch_geometric.nn as pyg_nn


class GCN_CONV(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCN_CONV, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)