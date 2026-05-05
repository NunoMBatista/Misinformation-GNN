import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool

class SimpleGNN(nn.Module):
    """
    A minimalistic Graph Convolutional Network (GCN) that takes 
    node features + edge_index, computes graph embeddings, 
    and predicts whether a thread is a rumour.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.5):
        super(SimpleGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.convs.append(GCNConv(in_dim, h_dim))
            in_dim = h_dim
            
        self.dropout_rate = dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Classifier head after pooling
        self.classifier = nn.Linear(in_dim, 1)
        
    def forward(self, x, edge_index, batch):
        """
        x: Node feature matrix [num_nodes_in_batch, input_dim]
        edge_index: Adjacency list [2, num_edges_in_batch]
        batch: Graph assignment vector for nodes [num_nodes_in_batch]
        """
        # 1. Message Passing (Node embeddings)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            
        # 2. Global Pooling (Graph embedding)
        # Averages node representations of each graph into a single graph vector
        x_graph = global_mean_pool(x, batch)
        
        # 3. Readout (Logits)
        out = self.classifier(x_graph)
        return out.squeeze(-1) # return raw logits


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for rumour classification.
    
    Unlike GCN, GAT uses attention mechanisms to weigh the importance 
    of neighboring tweets differently. We use GATv2Conv which is a 
    more robust version of the original GAT layer.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], heads=4, dropout=0.5):
        super(GATModel, self).__init__()
        
        self.convs = nn.ModuleList()
        in_dim = input_dim
        
        # We'll use multi-head attention to capture different types of 
        # interaction dynamics (e.g., skeptical vs. supportive).
        for h_dim in hidden_dims:
            # GATv2Conv output dimension is h_dim * heads
            self.convs.append(GATv2Conv(in_dim, h_dim, heads=heads, dropout=dropout))
            in_dim = h_dim * heads # Update in_dim for the next layer
            
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier head after pooling the multi-head outputs
        self.classifier = nn.Linear(in_dim, 1)
        
    def forward(self, x, edge_index, batch):
        """
        x: Node feature matrix
        edge_index: Adjacency list
        batch: Graph assignment vector
        """
        # 1. Attention-based Message Passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            
        # 2. Global Pooling
        # Collapses the graph into a single vector by averaging node embeddings
        x_graph = global_mean_pool(x, batch)
        
        # 3. Classification
        out = self.classifier(x_graph)
        return out.squeeze(-1)
