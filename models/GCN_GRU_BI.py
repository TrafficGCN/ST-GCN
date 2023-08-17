# Author 2023 Thomas Fink

import torch
import torch.nn as nn

from models.GCN_CONV import GCN_CONV

class GCN_GRU_BI(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0):
        super(GCN_GRU_BI, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.layers.append(GCN_CONV(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))

        for _ in range(num_gcn_layers - 2):
            self.layers.append(GCN_CONV(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))

        self.gru = nn.GRU(hidden_channels, hidden_channels, num_layers=num_rnn_layers,  batch_first=True, bidirectional=True)

        # Double the hidden_channels for output layer since bidirectional GRU returns double the size of hidden units.
        self.output_layer = nn.Linear(hidden_channels*2, num_predictions)

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        x, _ = self.gru(x.unsqueeze(0))
        x = x.squeeze(0)

        x = self.output_layer(x)

        return x.squeeze()