import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss with in-batch negatives.
    temperature=0.05 by default.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor) -> torch.Tensor:
        anchor_emb = F.normalize(anchor_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)

        sim = torch.matmul(anchor_emb, positive_emb.T) / self.temperature

        batch_size = anchor_emb.size(0)
        labels = torch.arange(batch_size, device=anchor_emb.device)

        return F.cross_entropy(sim, labels)
