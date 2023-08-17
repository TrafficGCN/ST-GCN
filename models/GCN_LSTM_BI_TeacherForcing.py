# Author 2023 Thomas Fink

import random
import torch
import torch.nn as nn

from models.GCN_LSTM_BI import GCN_LSTM_BI

class GCN_LSTM_BI_TeacherForcing(GCN_LSTM_BI):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout=0, teacher_forcing_ratio=0.25):
        super(GCN_LSTM_BI_TeacherForcing, self).__init__(in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x, edge_index, edge_weight, target=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        # BiLSTM expects 3D input (seq_len, batch, input_size)
        # If x is 2D, we use unsqueeze to add an extra dimension
        x = x.unsqueeze(0) if len(x.shape) == 2 else x

        if target is not None:
            predictions = []
            input = x

            for i in range(target.size(1)):  # Assuming target is (batch_size, seq_len, feature_size)
                output, _ = self.bilstm(input)
                output = self.output_layer(output).squeeze()
                predictions.append(output)

                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                input = target[:, i, :].unsqueeze(0) if use_teacher_forcing else output.unsqueeze(0)

            x = torch.stack(predictions, dim=1)
        else:
            x, _ = self.bilstm(x)
            # x shape after LSTM: (seq_len, batch, num_directions * hidden_size)
            # Correct the view operation:
            x = x.view(x.size(0), -1, self.hidden_channels * self.num_directions)
            x = self.output_layer(x)

        return x.squeeze()