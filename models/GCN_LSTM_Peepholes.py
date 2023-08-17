# Author 2023 Thomas Fink

import torch
import torch.nn as nn

from models.GCN_CONV import GCN_CONV


class LSTM_CellWithPeepholes(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_CellWithPeepholes, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.weight_ch = nn.Parameter(torch.Tensor(hidden_size, 3 * hidden_size))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))

    def forward(self, input, state):
        hx, cx = state
        gates = torch.mm(input, self.weight_ih) + torch.mm(hx, self.weight_hh) + self.bias

        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        cell_gate = cell_gate + torch.mm(cx, self.weight_ch[:, :self.hidden_size])
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate + torch.mm(cx, self.weight_ch[:, self.hidden_size:2*self.hidden_size]))
        cell_gate = torch.tanh(cell_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        out_gate = out_gate + torch.mm(cy, self.weight_ch[:, -self.hidden_size:])
        hy = torch.sigmoid(out_gate) * torch.tanh(cy)

        return hy, cy

class GCN_LSTM_Peepholes(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0):
        super(GCN_LSTM_Peepholes, self).__init__()
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

        self.rnn = nn.ModuleList()
        for _ in range(num_rnn_layers):
            self.rnn.append(LSTM_CellWithPeepholes(hidden_channels, hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, num_predictions)

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        if x.dim() == 2:  # Add sequence length dimension if it's missing
            x = x.unsqueeze(1)

        h_t, c_t = torch.zeros(x.size(0), self.rnn[0].hidden_size).to(x.device), torch.zeros(x.size(0), self.rnn[0].hidden_size).to(x.device)
        for t in range(x.size(1)):  # Iterate over sequence
            for rnn_cell in self.rnn:
                h_t, c_t = rnn_cell(x[:, t, :], (h_t, c_t))  # Pass 2D tensor to LSTM cell
        x = h_t

        x = self.output_layer(x)

        return x.squeeze()
