# Author 2023 Thomas Fink

import random
import torch

from models.GCN_GRU import GCN_GRU

class GCN_GRU_TeacherForcing(GCN_GRU):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, teacher_forcing_ratio=0.5, dropout=0):
        super(GCN_GRU_TeacherForcing, self).__init__(in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_predictions, dropout)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x, edge_index, edge_weight, target=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        x = x.unsqueeze(0)
        
        if self.training and target is not None:
            outputs = []
            hidden = None
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            for i in range(target.size(1)):  # target size = (batch_size, seq_len, input_size)
                if use_teacher_forcing:
                    # Use target as input
                    out, hidden = self.gru(target[:, i].unsqueeze(1), hidden)
                    out = self.output_layer(out)
                    outputs.append(out)
                else:
                    # Use model's own prediction as input
                    out, hidden = self.gru(out, hidden)
                    out = self.output_layer(out)
                    outputs.append(out)
            x = torch.cat(outputs, dim=1)
        else:
            x, _ = self.gru(x)
            x = self.output_layer(x)

        return x.squeeze()