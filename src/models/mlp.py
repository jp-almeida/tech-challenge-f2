"""PyTorch MLP model for recommendation with user/item embeddings."""
import torch
import torch.nn as nn


class RecommenderMLP(nn.Module):
    """MLP com embeddings de usuário/item, BatchNorm e Dropout.

    A entrada esperada é um tensor onde:
        coluna 0 -> user_idx (índice contíguo do usuário)
        coluna 1 -> item_idx (índice contíguo do item)
        colunas 2: -> features contínuas já escalonadas (StandardScaler)

    IMPORTANTE: a camada final NÃO aplica mais Sigmoid. O modelo retorna
    logits e deve ser treinado com nn.BCEWithLogitsLoss (implementação
    recomendada pelo PyTorch, mais estável numericamente do que
    Sigmoid + BCELoss). Para obter probabilidades na inferência, aplique
    torch.sigmoid(model(x)).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_continuous_features: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the MLP.

        Args:
            num_users (int): Número total de usuários únicos (dimensiona o embedding).
            num_items (int): Número total de itens únicos (dimensiona o embedding).
            num_continuous_features (int): Número de features contínuas (não-id).
            embedding_dim (int): Tamanho dos vetores de embedding de usuário/item.
            hidden_dim (int): Número de unidades na primeira camada oculta.
            dropout (float): Probabilidade de dropout.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2 + num_continuous_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Tensor de entrada. Coluna 0 = user_idx,
                coluna 1 = item_idx, colunas 2: = features contínuas.

        Returns:
            torch.Tensor: Logits (sem Sigmoid aplicado).
        """
        user_idx = x[:, 0].long()
        item_idx = x[:, 1].long()
        continuous_features = x[:, 2:]

        user_vector = self.user_embedding(user_idx)
        item_vector = self.item_embedding(item_idx)

        combined = torch.cat([user_vector, item_vector, continuous_features], dim=1)

        return self.net(combined)