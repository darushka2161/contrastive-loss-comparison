import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    """
    MSE(cosine_similarity(s1, s2), target_score).
    Target scores: entailment=1.0, neutral=0.5, contradiction=0.0.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        cosine_sim = (emb1 * emb2).sum(dim=-1)
        return self.mse(cosine_sim, target)
