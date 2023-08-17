# Author 2023 Thomas Fink

import torch
import torch.nn as nn

class AttentionLayer_GRU_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionLayer_GRU_LSTM, self).__init__()
        self.linear_query = nn.Linear(in_channels, out_channels)
        self.linear_key = nn.Linear(in_channels, out_channels)
        self.linear_value = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = self.softmax(attention_scores)

        x = torch.matmul(attention_scores, value)
        return x