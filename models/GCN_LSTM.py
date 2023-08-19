# Author 2023 Thomas Fink

import torch
import torch.nn as nn

from models.GCN_CONV import GCN_CONV
from models.AttentionLayer_GRU_LSTM import AttentionLayer_GRU_LSTM

class GCN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0):
        super(GCN_LSTM, self).__init__()
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

        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_rnn_layers,  batch_first=True)

        # Add attention layer
        self.attention_layer = AttentionLayer_GRU_LSTM(hidden_channels, hidden_channels)

        self.output_layer = nn.Linear(hidden_channels, num_predictions)

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        x, _ = self.lstm(x.unsqueeze(0))
        x = x.squeeze(0)

        # Use the attention layer
        #x = self.attention_layer(x)

        x = self.output_layer(x)

        return x.squeeze()