"""Metrics shared by training and evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    root_mean_squared_error,
)


def calculate_binary_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "rmse": float(root_mean_squared_error(target, probabilities)),
    }


def calculate_ranking_metrics(
    user_ids: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Compute Precision@K, Recall@K, NDCG@K and MAP@K (Fase 10).

    Diferente das métricas de classificação, essas métricas avaliam a
    qualidade do RANKING de itens recomendados por usuário, que é o que
    realmente importa em sistemas de recomendação (em vez de tratar cada
    par usuário-item de forma independente).

    Args:
        user_ids: índice do usuário (user_idx) para cada linha do conjunto
            de teste.
        target: rótulo binário de relevância (1 = item relevante).
        probabilities: score/probabilidade previsto pelo modelo.
        k: tamanho do corte do ranking (top-K).
    """

    precisions, recalls, ndcgs, average_precisions = [], [], [], []

    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        user_target = target[mask]
        user_scores = probabilities[mask]

        if user_target.sum() == 0:
            continue

        order = np.argsort(-user_scores)
        ranked_target = user_target[order][:k]

        num_relevant_at_k = ranked_target.sum()
        num_relevant_total = user_target.sum()

        precisions.append(num_relevant_at_k / k)
        recalls.append(num_relevant_at_k / num_relevant_total)

        discounts = 1.0 / np.log2(np.arange(2, len(ranked_target) + 2))
        dcg = (ranked_target * discounts).sum()
        ideal_ranked = np.sort(user_target)[::-1][:k]
        idcg = (ideal_ranked * discounts[: len(ideal_ranked)]).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        hits, precision_sum = 0, 0.0
        for rank, is_relevant in enumerate(ranked_target, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(
            precision_sum / num_relevant_total if num_relevant_total > 0 else 0.0
        )

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map_at_{k}": float(np.mean(average_precisions)) if average_precisions else 0.0,
    }