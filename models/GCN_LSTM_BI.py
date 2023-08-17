# Author 2023 Thomas Fink

import torch
import torch.nn as nn

from models.GCN_CONV import GCN_CONV

class GCN_LSTM_BI(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0):
        super(GCN_LSTM_BI, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_rnn_layers = num_rnn_layers
        self.num_directions = 2  # For bidirectional LSTM

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

        self.bilstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_rnn_layers,  batch_first=True, bidirectional=True)

        self.output_layer = nn.Linear(hidden_channels * self.num_directions, num_predictions)

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        # BiLSTM expects 3D input (seq_len, batch, input_size)
        # If x is 2D, we use unsqueeze to add an extra dimension
        x = x.unsqueeze(0) if len(x.shape) == 2 else x

        x, _ = self.bilstm(x)

        # x shape after LSTM: (seq_len, batch, num_directions * hidden_size)
        # Correct the view operation:
        x = x.view(x.size(0), -1, self.hidden_channels * self.num_directions)


        x = self.output_layer(x)

        return x.squeeze()