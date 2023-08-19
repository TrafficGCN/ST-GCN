# Author 2023 Thomas Fink

import torch
import torch.nn as nn

'''
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
'''

class AttentionLayer_GRU_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads=1):
        super(AttentionLayer_GRU_LSTM, self).__init__()
        self.n_heads = n_heads
        self.head_dim = out_channels // n_heads

        # Make sure that the embedding dimension of model is a multiple of number of heads.
        assert (
            self.head_dim * n_heads == out_channels
        ), "out_channels must be divisible by n_heads."

        self.linear_query = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_key = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_value = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_out = nn.Linear(out_channels, out_channels, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.linear_query(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.linear_key(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.linear_value(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)  # Scale the attention scores
        attention_scores = self.softmax(attention_scores)

        attention_output = torch.matmul(attention_scores, value)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)

        # Pass through output linear layer
        x = self.linear_out(attention_output)

        return x
