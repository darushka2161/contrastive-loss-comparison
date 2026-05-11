import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple


class NLIPairDataset(Dataset):
    """Dataset for InfoNCE and similar pair-based losses."""

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, positive = self.pairs[idx]
        return {"anchor": anchor, "positive": positive}


class NLITripletDataset(Dataset):
    """Dataset for Triplet loss: (anchor, positive, negative)."""

    def __init__(self, triplets: List[Tuple[str, str, str]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return {"anchor": anchor, "positive": positive, "negative": negative}


class NLICosineDataset(Dataset):
    """Dataset for Cosine Similarity loss: (s1, s2, score)."""

    def __init__(self, samples: List[Tuple[str, str, float]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s1, s2, score = self.samples[idx]
        return {"sentence1": s1, "sentence2": s2, "score": score}


class STSDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pair(batch, tokenizer, max_length=128):
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]
    enc_a = tokenizer(anchors, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_p = tokenizer(positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc_a, enc_p


def collate_triplet(batch, tokenizer, max_length=128):
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = [b["negative"] for b in batch]
    enc_a = tokenizer(anchors, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_p = tokenizer(positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_n = tokenizer(negatives, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc_a, enc_p, enc_n


def collate_cosine(batch, tokenizer, max_length=128):
    s1 = [b["sentence1"] for b in batch]
    s2 = [b["sentence2"] for b in batch]
    scores = torch.tensor([b["score"] for b in batch], dtype=torch.float)
    enc1 = tokenizer(s1, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc2 = tokenizer(s2, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return enc1, enc2, scores
