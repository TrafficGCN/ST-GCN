import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GCN_CONV import GCN_CONV

class Attention(nn.Module):
    def __init__(self, hidden_dim, scale=True):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_linear = nn.Linear(hidden_dim, 1)
        self.scale = scale

    def forward(self, inputs):
        raw_weights = self.attention_linear(inputs)
        if self.scale:
            raw_weights /= self.hidden_dim**0.5
        attention_weights = F.softmax(raw_weights, dim=1)
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
        # Temporal Attention
        temporal_attentions = []
        for i in range(1, self.num_lags + 1):
            lagged_x = x[:, :-i]
            current_x = x[:, i:]
            attention = torch.sum(self.temporal_weights[i-1] * lagged_x * current_x, dim=2, keepdim=True)
            temporal_attentions.append(attention)
        
        temporal_attention = torch.cat(temporal_attentions, dim=1)
        
        # Spatial Attention
        spatial_attention = torch.sum(self.spatial_weights * x[:, None, :] * spatial_neighbors, dim=2, keepdim=True)
        
        # Combining the attentions
        combined_attention = temporal_attention + spatial_attention
        attention_weights = F.softmax(combined_attention, dim=1)
        
        # Weighted sum of features
        output = torch.sum(attention_weights * x, dim=1)
        
        return output

class GCN_LSTM_BI_Multi_Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0, activation="relu", num_lags=5):
        super(GCN_LSTM_BI_Multi_Attention, self).__init__()
        assert activation in ["relu", "tanh", "sigmoid"], "Invalid activation function"
        
        self.activation_fn = nn.ReLU() if activation == "relu" else (nn.Tanh() if activation == "tanh" else nn.Sigmoid())
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
        self.attention = Attention(hidden_channels * self.num_directions)
        self.spatiotemporal_attention = SpatioTemporalAttention(hidden_channels * self.num_directions, num_lags)

        self.output_layer = nn.Linear(hidden_channels * self.num_directions, num_predictions)

    def forward(self, x, edge_index, edge_weight, spatial_neighbors=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = self.activation_fn(x)
            x = self.dropouts[i](x)

        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        x, _ = self.bilstm(x)
        x = x.view(x.size(1), -1, self.hidden_channels * self.num_directions)

        # Use spatio-temporal attention if spatial_neighbors is provided
        if spatial_neighbors is not None:
            x = self.spatiotemporal_attention(x, spatial_neighbors)
        else:
            x, _ = self.attention(x)

        x = self.output_layer(x)
        return x.squeeze()

