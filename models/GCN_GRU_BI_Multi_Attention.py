# Author 2023 Thomas Fink

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GCN_CONV import GCN_CONV

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        attention_weights = F.softmax(self.attention_linear(inputs), dim=1)
        weighted = torch.mul(inputs, attention_weights)
        outputs = weighted.sum(1)
        return outputs, attention_weights

class SpatioTemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_lags):
        super(SpatioTemporalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_lags = num_lags
        self.temporal_weights = nn.Parameter(torch.randn(num_lags, 1, 1))
        self.spatial_weights = nn.Parameter(torch.randn(feature_dim, 1, 1))

    def forward(self, x, spatial_neighbors):
        temporal_attentions = []
        for i in range(1, self.num_lags + 1):
            lagged_x = x[:, :-i]
            current_x = x[:, i:]
            attention = torch.sum(self.temporal_weights[i-1] * lagged_x * current_x, dim=2, keepdim=True)
            temporal_attentions.append(attention)

        temporal_attention = torch.cat(temporal_attentions, dim=1)
        spatial_attention = torch.sum(self.spatial_weights * x[:, None, :] * spatial_neighbors, dim=2, keepdim=True)
        combined_attention = temporal_attention + spatial_attention
        attention_weights = F.softmax(combined_attention, dim=1)
        output = torch.sum(attention_weights * x, dim=1)

        return output

class GCN_GRU_BI_Multi_Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0, num_lags=5):
        super(GCN_GRU_BI_Multi_Attention, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_rnn_layers = num_rnn_layers
        self.num_directions = 2

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

        self.bigru = nn.GRU(hidden_channels, hidden_channels, num_layers=num_rnn_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_channels * self.num_directions)
        self.spatiotemporal_attention = SpatioTemporalAttention(hidden_channels * self.num_directions, num_lags)
        self.output_layer = nn.Linear(hidden_channels * self.num_directions, num_predictions)

    def forward(self, x, edge_index, edge_weight, spatial_neighbors=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        x, _ = self.bigru(x)
        x = x.view(x.size(1), -1, self.hidden_channels * self.num_directions)

        if spatial_neighbors is not None:
            x = self.spatiotemporal_attention(x, spatial_neighbors)
        else:
            x, _ = self.attention(x)

        x = self.output_layer(x)

        return x.squeeze()
