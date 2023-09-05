import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

import torch.nn as nn
import torch_geometric.nn as pyg_nn

from torch_geometric.nn import GATConv


'''
class ImprovedGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6, 
                 activation=None, concat=True, use_residual=False, use_skip=False):
        super(ImprovedGATLayer, self).__init__()
        
        if use_residual and in_channels != out_channels * heads:
            raise ValueError("For residual connections, in_channels should be equal to out_channels * heads.")
        
        self.use_residual = use_residual
        self.use_skip = use_skip
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.activation = activation or nn.ReLU()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels * heads if concat else out_channels)
        
    def forward(self, x, edge_index):
        out = self.gat_conv(x, edge_index)
        
        # Residual connection
        if self.use_residual:
            out += x
            
        # Skip connection
        if self.use_skip:
            out = torch.cat([x, out], dim=-1)
        
        # Apply Layer normalization
        out = self.layer_norm(out)
        
        return self.activation(out)



class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6, activation=None):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=dropout)
        self.activation = activation or nn.ReLU()

    def forward(self, x, edge_index):
        return self.activation(self.gat_conv(x, edge_index))

class GCN_CONV(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, activation=None, use_batch_norm=False, 
                 residual=False, use_attention=False, heads=1, dropout=0.6):
        super(GCN_CONV, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        self.activation = activation or nn.ReLU()
        self.residual = residual
        self.use_attention = use_attention

        if self.residual and in_channels != out_channels:
            raise ValueError("For residual connections, in_channels and out_channels must be the same.")
        
        if self.use_attention:
            self.attention_layer = ImprovedGATLayer(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        if self.use_attention:
            out = self.attention_layer(x, edge_index)
        else:
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
'''


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