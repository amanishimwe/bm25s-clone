"""Numba-accelerated selection (stub)."""

import numpy as np

def topk(scores: np.ndarray, k: int, sorted: bool = True, backend: str = "numba"):
    """Numba top-k selection."""
    if sorted:
        indices = np.argsort(-scores)[:k]
        topk_scores = scores[indices]
    else:
        indices = np.argsort(-scores)[:k]
        topk_scores = scores[indices]
    
    return topk_scores, indices
