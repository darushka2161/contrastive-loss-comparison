import torch
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm

from training.metrics import compute_spearman, compute_embedding_stats, compute_cosine_distribution


def evaluate_on_sts(model, tokenizer, sts_samples: List[Dict], device,
                    batch_size: int = 64, max_length: int = 128) -> float:
    model.eval()
    s1_list = [s["sentence1"] for s in sts_samples]
    s2_list = [s["sentence2"] for s in sts_samples]
    gold_scores = [s["score"] for s in sts_samples]

    emb1 = model.encode_sentences(s1_list, tokenizer, batch_size, max_length, str(device))
    emb2 = model.encode_sentences(s2_list, tokenizer, batch_size, max_length, str(device))

    spearman = compute_spearman(emb1, emb2, gold_scores)
    model.train()
    return spearman


def detailed_evaluation(model, tokenizer, sts_samples: List[Dict], device,
                        batch_size: int = 64, max_length: int = 128) -> Dict:
    model.eval()
    s1_list = [s["sentence1"] for s in sts_samples]
    s2_list = [s["sentence2"] for s in sts_samples]
    gold_scores = [s["score"] for s in sts_samples]

    emb1 = model.encode_sentences(s1_list, tokenizer, batch_size, max_length, str(device))
    emb2 = model.encode_sentences(s2_list, tokenizer, batch_size, max_length, str(device))

    spearman = compute_spearman(emb1, emb2, gold_scores)
    emb_stats = compute_embedding_stats(torch.cat([emb1, emb2], dim=0))
    cosine_stats = compute_cosine_distribution(emb1, emb2)

    model.train()
    return {
        "spearman": spearman,
        "embedding_stats": emb_stats,
        "cosine_distribution": cosine_stats,
    }
