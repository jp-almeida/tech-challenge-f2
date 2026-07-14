"""PyTorch training routines with early stopping."""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim


def train_mlp(model: nn.Module, features: torch.Tensor, target: torch.Tensor) -> tuple[nn.Module, float]:
    """Train an MLP and restore its best loss checkpoint."""
    optimizer, criterion = optim.Adam(model.parameters(), lr=0.01), nn.BCELoss()
    best_loss, best_state, patience = float("inf"), None, 5
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(features), target)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss, best_state, patience = loss.item(), deepcopy(model.state_dict()), 5
        else:
            patience -= 1
        if patience == 0:
            break
    model.load_state_dict(best_state)
    return model, best_loss
