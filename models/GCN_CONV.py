import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GCN_CONV(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, activation=None, use_batch_norm=False, residual=False):
        super(GCN_CONV, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        self.activation = activation or nn.ReLU()
        self.residual = residual
        
        # For residual connection, the in_channels and out_channels must be same
        if self.residual and in_channels != out_channels:
            raise ValueError("For residual connections, in_channels and out_channels must be the same.")

    def forward(self, x, edge_index, edge_weight):
        out = self.linear(x)
        
        # Residual connection
        if self.residual:
            out += x
            
        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
        
        if self.use_batch_norm:
            out = self.batch_norm(out)
            
        return self.activation(out)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

# gcn_layer = GCN_CONV(in_channels=16, out_channels=16, activation=nn.LeakyReLU(), use_batch_norm=True, residual=True)

'''
class GCN_CONV(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCN_CONV, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)
'''