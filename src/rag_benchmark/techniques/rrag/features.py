from __future__ import annotations

import numpy as np


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def compute_retrieval_features(query_vector, passage_vectors, top_k: int = 10):
    """Return official R²AG feature rows and source indices in relevance order."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    passages = np.asarray(passage_vectors, dtype=np.float32)
    if passages.ndim != 2 or not len(passages):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    scores = np.asarray([_cosine(query, row) for row in passages], dtype=np.float32)
    order = np.argsort(scores)[::-1][:top_k]
    ranked = passages[order]
    ranked_scores = scores[order]

    neighbor: list[float] = []
    if len(ranked) == 1:
        neighbor = [1.0]
    else:
        for index, row in enumerate(ranked):
            adjacent = []
            if index:
                adjacent.append(_cosine(row, ranked[index - 1]))
            if index + 1 < len(ranked):
                adjacent.append(_cosine(row, ranked[index + 1]))
            neighbor.append(float(np.mean(adjacent)))

    precedent = [1.0]
    for index in range(1, len(ranked)):
        previous_scores = ranked_scores[:index]
        weights = np.exp(previous_scores - np.max(previous_scores))
        weights = weights / weights.sum()
        aggregate = np.sum(ranked[:index] * weights[:, None], axis=0)
        precedent.append(_cosine(ranked[index], aggregate))

    features = np.column_stack((ranked_scores, neighbor, precedent)).astype(np.float32)
    return features, order.astype(np.int64)

