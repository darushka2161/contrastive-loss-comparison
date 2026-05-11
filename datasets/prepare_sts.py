import os
import pickle
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm


def _load_sts_split(split: str, cache_dir: str) -> List[Dict]:
    """Load one STS Benchmark split and normalize scores to [0, 1]."""
    print(f"Loading STS Benchmark ({split})...")
    dataset = load_dataset("mteb/stsbenchmark-sts", cache_dir=cache_dir, split=split)
    samples = []
    for ex in tqdm(dataset, desc=f"Processing STS {split}"):
        s1 = ex["sentence1"].strip()
        s2 = ex["sentence2"].strip()
        score = float(ex["score"]) / 5.0
        if s1 and s2:
            samples.append({"sentence1": s1, "sentence2": s2, "score": score})
    print(f"  Loaded {len(samples)} samples")
    return samples


def load_sts_benchmark(split: str = "test", cache_dir: str = "data/cache") -> List[Dict]:
    return _load_sts_split(split, cache_dir)


def load_sts_as_cosine_dataset(split: str = "train",
                                cache_dir: str = "data/cache") -> List[Tuple[str, str, float]]:
    """Load STS split as (s1, s2, score) tuples for CosineSimilarityLoss training."""
    samples = _load_sts_split(split, cache_dir)
    return [(s["sentence1"], s["sentence2"], s["score"]) for s in samples]


def prepare_sts_and_save(processed_dir: str = "data/processed", cache_dir: str = "data/cache"):
    os.makedirs(processed_dir, exist_ok=True)

    # STS test — only for final evaluation, never used during training/tuning
    test_samples = _load_sts_split("test", cache_dir)
    with open(os.path.join(processed_dir, "sts_test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    # STS validation (dev) — used for validation during training
    try:
        val_samples = _load_sts_split("validation", cache_dir)
        with open(os.path.join(processed_dir, "sts_val.pkl"), "wb") as f:
            pickle.dump(val_samples, f)
    except Exception:
        val_samples = test_samples
        with open(os.path.join(processed_dir, "sts_val.pkl"), "wb") as f:
            pickle.dump(val_samples, f)

    # STS train — for Stage 2 fine-tuning with human similarity annotations
    try:
        train_samples = _load_sts_split("train", cache_dir)
        with open(os.path.join(processed_dir, "sts_train.pkl"), "wb") as f:
            pickle.dump(train_samples, f)
        # Also save as cosine tuples for direct use in NLICosineDataset
        train_tuples = [(s["sentence1"], s["sentence2"], s["score"]) for s in train_samples]
        with open(os.path.join(processed_dir, "sts_train_cosine.pkl"), "wb") as f:
            pickle.dump(train_tuples, f)
    except Exception as e:
        print(f"Warning: could not load STS train split: {e}")
        train_samples = []

    return test_samples, val_samples, train_samples


def load_sts_processed(processed_dir: str = "data/processed", split: str = "test") -> List[Dict]:
    """
    Load processed STS samples.
    - split='test'  → final evaluation only
    - split='val'   → validation during training
    - split='train' → Stage 2 fine-tuning data
    """
    path = os.path.join(processed_dir, f"sts_{split}.pkl")
    if not os.path.exists(path):
        fallback = os.path.join(processed_dir, "sts_test.pkl")
        print(f"Warning: {path} not found, falling back to sts_test.pkl")
        path = fallback
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sts_cosine_tuples(processed_dir: str = "data/processed") -> List[Tuple[str, str, float]]:
    """Load STS train split as (s1, s2, score) for CosineSimilarityLoss fine-tuning."""
    path = os.path.join(processed_dir, "sts_train_cosine.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
