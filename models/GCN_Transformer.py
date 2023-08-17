# Author 2023 Thomas Fink

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.GCN_CONV import GCN_CONV

class GCN_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_transformer_layers, num_predictions, dropout=0):
        super(GCN_Transformer, self).__init__()
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

        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=hidden_channels, nhead=1) # change nhead based on your problem
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=num_transformer_layers)

        self.output_layer = nn.Linear(hidden_channels, num_predictions)

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)

        x = self.output_layer(x)

        return x.squeeze()