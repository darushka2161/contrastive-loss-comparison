import os
import pickle
import random
from typing import List, Tuple
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

LABEL_ENTAILMENT = 0
LABEL_NEUTRAL = 1
LABEL_CONTRADICTION = 2

SNLI_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _get_snli_label(example):
    return example.get("label", -1)


def load_nli_raw(cache_dir: str = "data/cache"):
    """Load and concatenate SNLI + MNLI training splits."""
    print("Loading SNLI...")
    snli = load_dataset("snli", cache_dir=cache_dir, split="train")
    snli = snli.filter(lambda x: x["label"] in [0, 1, 2])

    print("Loading MNLI...")
    mnli = load_dataset("multi_nli", cache_dir=cache_dir, split="train")
    mnli = mnli.rename_column("premise", "sentence1").rename_column("hypothesis", "sentence2")
    mnli = mnli.select_columns(["sentence1", "sentence2", "label"])
    mnli = mnli.filter(lambda x: x["label"] in [0, 1, 2])

    snli = snli.rename_column("premise", "sentence1").rename_column("hypothesis", "sentence2")
    snli = snli.select_columns(["sentence1", "sentence2", "label"])

    combined = concatenate_datasets([snli, mnli])
    print(f"Combined NLI corpus: {len(combined)} examples")
    return combined


def build_pair_dataset(dataset, max_samples: int = None) -> List[Tuple[str, str]]:
    """Build (anchor, positive) pairs from entailment examples."""
    pairs = []
    for ex in tqdm(dataset, desc="Building pairs"):
        if ex["label"] == LABEL_ENTAILMENT:
            s1 = ex["sentence1"].strip()
            s2 = ex["sentence2"].strip()
            if s1 and s2:
                pairs.append((s1, s2))
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs


def build_triplet_dataset(dataset, max_samples: int = None) -> List[Tuple[str, str, str]]:
    """Build (anchor, positive, negative) triplets."""
    entailment = []
    contradiction = []

    for ex in tqdm(dataset, desc="Collecting NLI examples"):
        s1, s2 = ex["sentence1"].strip(), ex["sentence2"].strip()
        if not s1 or not s2:
            continue
        if ex["label"] == LABEL_ENTAILMENT:
            entailment.append((s1, s2))
        elif ex["label"] == LABEL_CONTRADICTION:
            contradiction.append((s1, s2))

    # Group by anchor sentence
    anchor_to_pos = {}
    anchor_to_neg = {}
    for s1, s2 in entailment:
        anchor_to_pos.setdefault(s1, []).append(s2)
    for s1, s2 in contradiction:
        anchor_to_neg.setdefault(s1, []).append(s2)

    triplets = []
    all_sentences = [s2 for _, s2 in entailment]
    random.shuffle(all_sentences)

    for anchor, pos_list in anchor_to_pos.items():
        neg_list = anchor_to_neg.get(anchor, [])
        negative = neg_list[0] if neg_list else random.choice(all_sentences)
        for positive in pos_list[:2]:
            triplets.append((anchor, positive, negative))
        if max_samples and len(triplets) >= max_samples:
            break

    random.shuffle(triplets)
    return triplets[:max_samples] if max_samples else triplets


def build_cosine_dataset(dataset, max_samples: int = None) -> List[Tuple[str, str, float]]:
    """Build (s1, s2, score) samples with scores: entailment=1.0, neutral=0.5, contradiction=0.0."""
    label_to_score = {
        LABEL_ENTAILMENT: 1.0,
        LABEL_NEUTRAL: 0.5,
        LABEL_CONTRADICTION: 0.0,
    }
    samples = []
    for ex in tqdm(dataset, desc="Building cosine samples"):
        label = ex["label"]
        if label not in label_to_score:
            continue
        s1, s2 = ex["sentence1"].strip(), ex["sentence2"].strip()
        if s1 and s2:
            samples.append((s1, s2, label_to_score[label]))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def prepare_and_save(processed_dir: str = "data/processed", cache_dir: str = "data/cache", max_samples: int = None):
    os.makedirs(processed_dir, exist_ok=True)
    dataset = load_nli_raw(cache_dir)

    print("Building pair dataset...")
    pairs = build_pair_dataset(dataset, max_samples)
    with open(os.path.join(processed_dir, "nli_pairs.pkl"), "wb") as f:
        pickle.dump(pairs, f)
    print(f"  Saved {len(pairs)} pairs")

    print("Building triplet dataset...")
    triplets = build_triplet_dataset(dataset, max_samples)
    with open(os.path.join(processed_dir, "nli_triplets.pkl"), "wb") as f:
        pickle.dump(triplets, f)
    print(f"  Saved {len(triplets)} triplets")

    print("Building cosine dataset...")
    cosine_samples = build_cosine_dataset(dataset, max_samples)
    with open(os.path.join(processed_dir, "nli_cosine.pkl"), "wb") as f:
        pickle.dump(cosine_samples, f)
    print(f"  Saved {len(cosine_samples)} cosine samples")

    return pairs, triplets, cosine_samples


def load_processed(processed_dir: str = "data/processed", loss_type: str = "info_nce"):
    if loss_type == "info_nce":
        path = os.path.join(processed_dir, "nli_pairs.pkl")
    elif loss_type == "triplet":
        path = os.path.join(processed_dir, "nli_triplets.pkl")
    elif loss_type == "cosine":
        path = os.path.join(processed_dir, "nli_cosine.pkl")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    with open(path, "rb") as f:
        return pickle.load(f)
