"""PyTorch MLP model for recommendation."""
import torch
import torch.nn as nn

class RecommenderMLP(nn.Module):
    """Simple MLP network for recommendations."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.net(x)
