import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.utils import to_undirected


def _root_indices(batch):
    """Index of the first node (root tweet) for each graph in a PyG batch."""
    prev = torch.cat([batch.new_full((1,), -1), batch[:-1]])
    return (batch != prev).nonzero(as_tuple=True)[0]


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


class ImprovedGNN(nn.Module):
    """
    GCN with three key fixes over SimpleGNN:
    1. Configurable edge direction (see edge_direction parameter).
    2. Root-readout — classifier always sees the source tweet's representation directly.
    3. Residual skip connections + LayerNorm — preserves node identity through aggregation.

    Classifier input: [global_mean_pool || root_node_repr]  (2 × last hidden dim)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.5, edge_direction='bidirectional'):
        super().__init__()
        self.edge_direction = edge_direction
        self.convs = nn.ModuleList()
        self.skips  = nn.ModuleList()
        self.norms  = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            self.convs.append(GCNConv(in_dim, h_dim))
            self.skips.append(nn.Linear(in_dim, h_dim, bias=False))
            self.norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim

        self.dropout_p  = dropout
        self.classifier = nn.Linear(in_dim * 2, 1)

    def forward(self, x, edge_index, batch):
        if self.edge_direction == 'bidirectional':
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        elif self.edge_direction == 'inverted':
            edge_index = edge_index.flip(0)

        for conv, skip, norm in zip(self.convs, self.skips, self.norms):
            x = norm(F.relu(conv(x, edge_index) + skip(x)))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_root = x[_root_indices(batch)]
        return self.classifier(torch.cat([x_mean, x_root], dim=-1)).squeeze(-1)


class ImprovedGAT(nn.Module):
    """
    GAT with the same three fixes applied to GATModel:
    1. Configurable edge direction (see edge_direction parameter).
    2. Root-readout concatenated to mean pool.
    3. Residual skip connections + LayerNorm.

    Classifier input: [global_mean_pool || root_node_repr]  (2 × last hidden dim)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], heads=2, dropout=0.5, edge_direction='bidirectional'):
        super().__init__()
        self.edge_direction = edge_direction
        self.convs = nn.ModuleList()
        self.skips  = nn.ModuleList()
        self.norms  = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            self.convs.append(GATv2Conv(in_dim, h_dim, heads=heads, dropout=dropout))
            out_dim = h_dim * heads
            self.skips.append(nn.Linear(in_dim, out_dim, bias=False))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.dropout_p  = dropout
        self.classifier = nn.Linear(in_dim * 2, 1)

    def forward(self, x, edge_index, batch):
        if self.edge_direction == 'bidirectional':
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        elif self.edge_direction == 'inverted':
            edge_index = edge_index.flip(0)

        for conv, skip, norm in zip(self.convs, self.skips, self.norms):
            x = norm(F.relu(conv(x, edge_index) + skip(x)))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_root = x[_root_indices(batch)]
        return self.classifier(torch.cat([x_mean, x_root], dim=-1)).squeeze(-1)
