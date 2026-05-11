import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_mean_pool
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


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (Xu et al. 2019) with root readout.

    Uses SUM aggregation, which is theoretically as expressive as the WL graph
    isomorphism test. Unlike GCN's mean, SUM preserves neighbour counts — letting
    the model distinguish "1 reply vs 10 replies", a direct proxy for virality.

    Classifier input: [global_mean_pool || root_node_repr]  (2 × last hidden dim)
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.5,
                 edge_direction='bidirectional'):
        super().__init__()
        self.edge_direction = edge_direction
        self.convs = nn.ModuleList()
        self.norms  = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            # Inner MLP: two-layer, as recommended in the GIN paper
            mlp = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim

        self.dropout_p  = dropout
        self.classifier = nn.Linear(in_dim * 2, 1)

    def forward(self, x, edge_index, batch):
        if self.edge_direction == 'bidirectional':
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        elif self.edge_direction == 'inverted':
            edge_index = edge_index.flip(0)

        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_root = x[_root_indices(batch)]
        return self.classifier(torch.cat([x_mean, x_root], dim=-1)).squeeze(-1)


class ImprovedGNN(nn.Module):
    """
    GCN with skip connections, root readout, and configurable edge direction.

    When edge_dim > 0, GCNConv is replaced with GATv2Conv(heads=1, edge_dim=edge_dim)
    so the model can consume edge features (cos-sim + reply latency).
    The classifier input is [global_mean_pool || root_node_repr] (2 × last hidden dim).
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.5,
                 edge_direction='bidirectional', edge_dim=0):
        super().__init__()
        self.edge_direction = edge_direction
        self.edge_dim       = edge_dim
        self.convs = nn.ModuleList()
        self.skips  = nn.ModuleList()
        self.norms  = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            if edge_dim > 0:
                # GATv2Conv(heads=1) output is h_dim — same shape as GCNConv
                self.convs.append(GATv2Conv(in_dim, h_dim, heads=1, edge_dim=edge_dim, dropout=dropout))
            else:
                self.convs.append(GCNConv(in_dim, h_dim))
            self.skips.append(nn.Linear(in_dim, h_dim, bias=False))
            self.norms.append(nn.LayerNorm(h_dim))
            in_dim = h_dim

        self.dropout_p  = dropout
        self.classifier = nn.Linear(in_dim * 2, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if self.edge_direction == 'bidirectional':
            if edge_attr is not None:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=x.size(0))
            else:
                edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        elif self.edge_direction == 'inverted':
            edge_index = edge_index.flip(0)

        for conv, skip, norm in zip(self.convs, self.skips, self.norms):
            if edge_attr is not None:
                conv_out = conv(x, edge_index, edge_attr=edge_attr)
            else:
                conv_out = conv(x, edge_index)
            x = norm(F.relu(conv_out + skip(x)))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_root = x[_root_indices(batch)]
        return self.classifier(torch.cat([x_mean, x_root], dim=-1)).squeeze(-1)


class ImprovedGAT(nn.Module):
    """
    GAT with skip connections, root readout, and configurable edge direction.

    When edge_dim > 0, GATv2Conv natively incorporates edge features into
    attention computation. Classifier input: [mean_pool || root] (2 × last hidden dim).
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], heads=2, dropout=0.5,
                 edge_direction='bidirectional', edge_dim=0):
        super().__init__()
        self.edge_direction = edge_direction
        self.edge_dim       = edge_dim
        self.convs = nn.ModuleList()
        self.skips  = nn.ModuleList()
        self.norms  = nn.ModuleList()

        in_dim = input_dim
        for h_dim in hidden_dims:
            self.convs.append(GATv2Conv(
                in_dim, h_dim, heads=heads, dropout=dropout,
                edge_dim=edge_dim if edge_dim > 0 else None,
            ))
            out_dim = h_dim * heads
            self.skips.append(nn.Linear(in_dim, out_dim, bias=False))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.dropout_p  = dropout
        self.classifier = nn.Linear(in_dim * 2, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if self.edge_direction == 'bidirectional':
            if edge_attr is not None:
                edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=x.size(0))
            else:
                edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        elif self.edge_direction == 'inverted':
            edge_index = edge_index.flip(0)

        for conv, skip, norm in zip(self.convs, self.skips, self.norms):
            if edge_attr is not None:
                conv_out = conv(x, edge_index, edge_attr=edge_attr)
            else:
                conv_out = conv(x, edge_index)
            x = norm(F.relu(conv_out + skip(x)))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_root = x[_root_indices(batch)]
        return self.classifier(torch.cat([x_mean, x_root], dim=-1)).squeeze(-1)
