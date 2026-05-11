import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

# Internal imports
from src.models import RandomForestBaseline, MLPBaseline, SimpleGNN, GATModel, GINModel, ImprovedGNN, ImprovedGAT


def focal_loss(logits, targets, gamma=2.0):
    """Binary focal loss — down-weights easy examples, focuses on hard ones."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def _balance_per_event(train_data, random_state=42):
    """Undersample the majority class within each event so every event
    contributes an equal number of rumour and non-rumour examples."""
    rng = random.Random(random_state)
    by_event_label = defaultdict(list)
    for d in train_data:
        by_event_label[(d.event, int(d.y.item()))].append(d)

    balanced = []
    events = sorted(set(event for event, _ in by_event_label))
    for event in events:
        cls0 = by_event_label[(event, 0)]
        cls1 = by_event_label[(event, 1)]
        n = min(len(cls0), len(cls1))
        if n == 0:
            continue
        balanced.extend(rng.sample(cls0, n))
        balanced.extend(rng.sample(cls1, n))

    rng.shuffle(balanced)
    return balanced


def train_rf(config, train_dataset, test_dataset):
    """Train and evaluate Random Forest on pre-pooled graph features."""
    rf = RandomForestBaseline(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 10),
        random_state=config.get('random_state', 42)
    )

    train_dataset = _balance_per_event(train_dataset, random_state=config.get('random_state', 42))

    # Graph-level pooling (mean)
    X_train = torch.stack([d.x.mean(dim=0) for d in tqdm(train_dataset, desc="Pooling Train", leave=False)]).numpy()
    y_train = torch.cat([d.y for d in train_dataset]).numpy()
    
    X_test = torch.stack([d.x.mean(dim=0) for d in tqdm(test_dataset, desc="Pooling Test", leave=False)]).numpy()
    y_test = torch.cat([d.y for d in test_dataset]).numpy()
    
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    
    return y_test.tolist(), preds.tolist()

def train_nn(config, model_name, train_dataset, test_dataset, input_dim, edge_dim=0):
    """Train and evaluate Neural Networks (MLP or GNN)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Device] Training {model_name.upper()} on: {device}")

    train_dataset = _balance_per_event(train_dataset, random_state=42)
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
            dropout=config.get('dropout', 0.5),
        ).to(device)
        is_gnn = True
    elif model_name == "gat":
        model = GATModel(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32]),
            heads=config.get('heads', 4),
            dropout=config.get('dropout', 0.5),
        ).to(device)
        is_gnn = True
    elif model_name == "gin":
        model = GINModel(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [128, 64]),
            dropout=config.get('dropout', 0.4),
            edge_direction=config.get('edge_direction', 'bidirectional'),
        ).to(device)
        is_gnn = True
    elif model_name == "improved_gnn":
        model = ImprovedGNN(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [128, 64]),
            dropout=config.get('dropout', 0.4),
            edge_direction=config.get('edge_direction', 'bidirectional'),
            edge_dim=edge_dim,
        ).to(device)
        is_gnn = True
    elif model_name == "improved_gat":
        model = ImprovedGAT(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [128, 64]),
            heads=config.get('heads', 2),
            dropout=config.get('dropout', 0.4),
            edge_direction=config.get('edge_direction', 'bidirectional'),
            edge_dim=edge_dim,
        ).to(device)
        is_gnn = True
    else:
        raise ValueError(f"Unknown neural network type: {model_name}")

    lr = float(config.get('learning_rate', 0.001))
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(config.get('weight_decay', 0.0001))
    )
    gamma = config.get('focal_gamma', 2.0)
    criterion = (lambda lo, tg: focal_loss(lo, tg, gamma=gamma)) if config.get('use_focal_loss', False) else nn.BCEWithLogitsLoss()

    epochs = config.get('epochs', 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    pbar = tqdm(range(epochs), desc=f"Training {model_name.upper()}", leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if is_gnn:
                ea = getattr(batch, 'edge_attr', None)
                if model_name in ['improved_gnn', 'improved_gat'] and ea is not None:
                    out = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
                else:
                    out = model(batch.x, batch.edge_index, batch.batch)
            else:
                x_pooled = global_mean_pool(batch.x, batch.batch)
                out = model(x_pooled)

            loss = criterion(out, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        pbar.set_postfix(loss=f"{epoch_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            if is_gnn:
                ea = getattr(batch, 'edge_attr', None)
                if model_name in ['improved_gnn', 'improved_gat'] and ea is not None:
                    out = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
                else:
                    out = model(batch.x, batch.edge_index, batch.batch)
            else:
                x_pooled = global_mean_pool(batch.x, batch.batch)
                out = model(x_pooled)

            preds = (torch.sigmoid(out) > 0.5).int()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
            
    return all_labels, all_preds
