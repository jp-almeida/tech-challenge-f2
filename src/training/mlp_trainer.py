"""PyTorch training routines with mini-batches, LR scheduling and early stopping."""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_mlp(
    model: nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[nn.Module, float]:
    """Train an MLP with mini-batches and restore its best loss checkpoint.

    Melhorias em relação à versão anterior:
    - Mini-batch via DataLoader, em vez de treinar tudo de uma vez.
    - Adam com weight_decay (regularização L2).
    - BCEWithLogitsLoss (o modelo agora retorna logits, sem Sigmoid).
    - ReduceLROnPlateau: reduz o learning rate quando a loss estagna
      (0.001 -> 0.0005 -> 0.0001 ...).
    - Early stopping de verdade: o treino é interrompido após `patience`
      épocas consecutivas sem melhora na loss (antes, o patience=5
      praticamente não era utilizado).
    """
    dataset = TensorDataset(features, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_target in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)

        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_loss