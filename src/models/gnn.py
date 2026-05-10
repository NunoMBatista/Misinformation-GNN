import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool


class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.convs.append(GCNConv(in_dim, h_dim))
            in_dim = h_dim

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        x_graph = global_mean_pool(x, batch)
        return self.classifier(x_graph).squeeze(-1)


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], heads=4, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.convs.append(GATv2Conv(in_dim, h_dim, heads=heads, dropout=dropout))
            in_dim = h_dim * heads   # GATv2Conv output dim is h_dim * heads

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        x_graph = global_mean_pool(x, batch)
        return self.classifier(x_graph).squeeze(-1)
