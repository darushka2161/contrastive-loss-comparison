import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from typing import List, Dict, Tuple


def compute_spearman(embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                     gold_scores: List[float]) -> float:
    e1 = F.normalize(embeddings1, dim=-1)
    e2 = F.normalize(embeddings2, dim=-1)
    cosine_scores = (e1 * e2).sum(dim=-1).cpu().numpy()
    gold = np.array(gold_scores)
    corr, _ = spearmanr(cosine_scores, gold)
    return float(corr)


def compute_embedding_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    norms = embeddings.norm(dim=-1).cpu().numpy()
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def compute_cosine_distribution(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> Dict[str, float]:
    e1 = F.normalize(embeddings1, dim=-1)
    e2 = F.normalize(embeddings2, dim=-1)
    sims = (e1 * e2).sum(dim=-1).cpu().numpy()
    return {
        "cosine_mean": float(sims.mean()),
        "cosine_std": float(sims.std()),
        "cosine_min": float(sims.min()),
        "cosine_max": float(sims.max()),
    }
