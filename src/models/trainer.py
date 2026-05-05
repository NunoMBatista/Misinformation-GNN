import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

# Internal imports
from src.models import RandomForestBaseline, MLPBaseline, SimpleGNN, GATModel

def train_rf(config, train_dataset, test_dataset):
    """Train and evaluate Random Forest on pre-pooled graph features."""
    rf = RandomForestBaseline(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 10),
        random_state=config.get('random_state', 42)
    )
    
    # Graph-level pooling (mean)
    X_train = torch.stack([d.x.mean(dim=0) for d in tqdm(train_dataset, desc="Pooling Train", leave=False)]).numpy()
    y_train = torch.cat([d.y for d in train_dataset]).numpy()
    
    X_test = torch.stack([d.x.mean(dim=0) for d in tqdm(test_dataset, desc="Pooling Test", leave=False)]).numpy()
    y_test = torch.cat([d.y for d in test_dataset]).numpy()
    
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    
    return y_test.tolist(), preds.tolist()

def train_nn(config, model_name, train_dataset, test_dataset, input_dim):
    """Train and evaluate Neural Networks (MLP or GNN)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Device] Training {model_name.upper()} on: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if model_name == "mlp":
        model = MLPBaseline(
            input_dim=input_dim, 
            hidden_dims=config.get('hidden_dims', [64, 32]), 
            dropout=config.get('dropout', 0.5)
        ).to(device)
        is_gnn = False
    elif model_name == "gnn":
        model = SimpleGNN(
            input_dim=input_dim, 
            hidden_dims=config.get('hidden_dims', [64, 32]), 
            dropout=config.get('dropout', 0.5)
        ).to(device)
        is_gnn = True
    elif model_name == "gat":
        model = GATModel(
            input_dim=input_dim, 
            hidden_dims=config.get('hidden_dims', [64, 32]),
            heads=config.get('heads', 4),
            dropout=config.get('dropout', 0.5)
        ).to(device)
        is_gnn = True
    else:
        raise ValueError(f"Unknown neural network type: {model_name}")

    optimizer = optim.Adam(
        model.parameters(), 
        lr=float(config.get('learning_rate', 0.001)), 
        weight_decay=float(config.get('weight_decay', 0.0001))
    )
    # y is [0] or [1], need float for BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = config.get('epochs', 50)
    pbar = tqdm(range(epochs), desc=f"Training {model_name.upper()}", leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if is_gnn:
                out = model(batch.x, batch.edge_index, batch.batch)
            else:
                x_pooled = global_mean_pool(batch.x, batch.batch)
                out = model(x_pooled)
                
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        pbar.set_postfix(loss=f"{epoch_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            if is_gnn:
                out = model(batch.x, batch.edge_index, batch.batch)
            else:
                x_pooled = global_mean_pool(batch.x, batch.batch)
                out = model(x_pooled)
                
            preds = (torch.sigmoid(out) > 0.5).int()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
            
    return all_labels, all_preds
