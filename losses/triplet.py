import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss with cosine distance and configurable margin.
    margin=0.3 by default.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_dist = 1.0 - (anchor * positive).sum(dim=-1)
        neg_dist = 1.0 - (anchor * negative).sum(dim=-1)

        return F.relu(pos_dist - neg_dist + self.margin).mean()
