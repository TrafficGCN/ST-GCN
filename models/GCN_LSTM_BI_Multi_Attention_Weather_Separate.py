# Author 2023 Thomas Fink

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GCN_CONV import GCN_CONV

class Attention(nn.Module):
    def __init__(self, hidden_channels, scale=True):
        super(Attention, self).__init__()
        self.hidden_channels = hidden_channels
        self.attention_linear = nn.Linear(hidden_channels, 1)
        self.scale = scale

    def forward(self, inputs):
        raw_weights = self.attention_linear(inputs)
        if self.scale:
            raw_weights /= self.hidden_channels**0.5
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



# New class: GCN_LSTM_BI_Multi_Attention_Weather_Separate
# Implementing the new class GCN_LSTM_BI_Multi_Attention_Weather_Separate with modifications

class GCN_LSTM_BI_Multi_Attention_Weather_Separate(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_predictions, num_gcn_layers, num_rnn_layers, num_lags, dropout, activation="relu"):
        super(GCN_LSTM_BI_Multi_Attention_Weather_Separate, self).__init__()

        # Assuming in_channels is the combined input of speed and temperature
        # We will split the input into two halves for speed and temperature
        in_channels //= 2  # Half the channels for each type of data

        self.layers_speed = nn.ModuleList()
        self.layers_temp = nn.ModuleList()

        self.batch_norms_speed = nn.ModuleList()
        self.batch_norms_temp = nn.ModuleList()

        self.dropouts_speed = nn.ModuleList()
        self.dropouts_temp = nn.ModuleList()

        # Layers for speed data
        self.layers_speed.append(GCN_CONV(in_channels, hidden_channels))
        self.batch_norms_speed.append(nn.BatchNorm1d(hidden_channels))
        self.dropouts_speed.append(nn.Dropout(dropout))

        # Layers for temperature data
        self.layers_temp.append(GCN_CONV(in_channels, hidden_channels))
        self.batch_norms_temp.append(nn.BatchNorm1d(hidden_channels))
        self.dropouts_temp.append(nn.Dropout(dropout))

        for _ in range(num_gcn_layers - 2):
            self.layers_speed.append(GCN_CONV(hidden_channels, hidden_channels))
            self.batch_norms_speed.append(nn.BatchNorm1d(hidden_channels))
            self.dropouts_speed.append(nn.Dropout(dropout))

            self.layers_temp.append(GCN_CONV(hidden_channels, hidden_channels))
            self.batch_norms_temp.append(nn.BatchNorm1d(hidden_channels))
            self.dropouts_temp.append(nn.Dropout(dropout))

        self.activation_fn = nn.ReLU() if activation == "relu" else (nn.Tanh() if activation == "tanh" else nn.Sigmoid())
        self.hidden_channels = hidden_channels
        self.num_rnn_layers = num_rnn_layers
        self.num_directions = 2  # For bidirectional LSTM

        self.bilstm_speed = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_rnn_layers, batch_first=True, bidirectional=True)
        self.bilstm_temp = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_rnn_layers, batch_first=True, bidirectional=True)
        
        combined_feature_size = 2 * hidden_channels * self.num_directions
        self.attention_speed = Attention(combined_feature_size)
        self.attention_temp = Attention(combined_feature_size)

        self.spatiotemporal_attention_speed = SpatioTemporalAttention(hidden_channels * self.num_directions, num_lags)
        self.spatiotemporal_attention_temp = SpatioTemporalAttention(hidden_channels * self.num_directions, num_lags)

        # Output layer after fusion
        self.output_layer = nn.Linear(2 * hidden_channels * self.num_directions, num_predictions)

    def forward(self, x, edge_index, edge_weight, spatial_neighbors=None):
        # Splitting the input into speed and temperature data
        x_speed = x[:, :x.size(1) // 2]
        x_temp = x[:, x.size(1) // 2:]

        # Processing speed data
        for i, layer in enumerate(self.layers_speed):
            x_speed = layer(x_speed, edge_index, edge_weight)
            x_speed = self.batch_norms_speed[i](x_speed)
            x_speed = self.activation_fn(x_speed)
            x_speed = self.dropouts_speed[i](x_speed)

        x_speed = x_speed.unsqueeze(0) if len(x_speed.shape) == 2 else x_speed
        x_speed, _ = self.bilstm_speed(x_speed)
        x_speed = x_speed.view(x_speed.size(1), -1, self.hidden_channels * self.num_directions)

        # Processing temperature data
        for i, layer in enumerate(self.layers_temp):
            x_temp = layer(x_temp, edge_index, edge_weight)
            x_temp = self.batch_norms_temp[i](x_temp)
            x_temp = self.activation_fn(x_temp)
            x_temp = self.dropouts_temp[i](x_temp)

        x_temp = x_temp.unsqueeze(0) if len(x_temp.shape) == 2 else x_temp
        x_temp, _ = self.bilstm_temp(x_temp)
        x_temp = x_temp.view(x_temp.size(1), -1, self.hidden_channels * self.num_directions)

        # Fusion of speed and temperature data
        x_combined = torch.cat((x_speed, x_temp), dim=2)

        # Use spatio-temporal attention if spatial_neighbors is provided
        if spatial_neighbors is not None:
            x_combined = self.spatiotemporal_attention_speed(x_combined, spatial_neighbors)
        else:
            x_combined, _ = self.attention_speed(x_combined)

        x_combined = self.output_layer(x_combined)
        return x_combined.squeeze()
