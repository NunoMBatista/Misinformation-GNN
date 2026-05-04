import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

class RandomForestBaseline:
    """
    A simple Random Forest wrapping class.
    We'll mean-pool node features into a single vector per graph manually before training.
    """
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            class_weight="balanced"
        )
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class MLPBaseline(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for baseline comparison.
    Will run on mean-pooled graph features.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.5):
        super(MLPBaseline, self).__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1)) # Binary classification (Rumour vs Non-Rumour)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # We assume X represents the already-pooled graphs [batch_size, feature_dim]
        out = self.network(x)
        return out.squeeze(-1) # return raw logits
