#!/usr/bin/env python3
"""Evaluate a trained model on STS Benchmark."""
import argparse
import json
import os
import torch

from config import load_config
from models.sentence_encoder import SentenceEncoder, get_tokenizer
from datasets.prepare_sts import load_sts_processed
from evaluation.sts_evaluator import detailed_evaluation
from evaluation.embedding_metrics import (
    get_embeddings_for_visualization, reduce_with_pca, reduce_with_tsne,
)
from evaluation.visualization import plot_embedding_projection
from training.train_utils import get_device, set_seed


SAMPLE_SENTENCES = [
    ("A man is playing a guitar.", "similar"),
    ("A musician strums an instrument.", "similar"),
    ("The dog runs across the field.", "dissimilar"),
    ("Children are playing in the park.", "dissimilar"),
    ("The weather is sunny today.", "unrelated"),
    ("Scientists discovered a new planet.", "unrelated"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--plot_embeddings", action="store_true", help="Generate embedding plots")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    tokenizer = get_tokenizer(cfg.model_name)
    model = SentenceEncoder(cfg.model_name, cfg.pooling).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sts_samples = load_sts_processed("data/processed", "test")
    results = detailed_evaluation(model, tokenizer, sts_samples, device)

    print(f"\n=== Evaluation Results ===")
    print(f"Spearman: {results['spearman']:.4f}")
    print(f"Embedding norm: {results['embedding_stats']['norm_mean']:.4f} ± {results['embedding_stats']['norm_std']:.4f}")
    print(f"Cosine sim: {results['cosine_distribution']['cosine_mean']:.4f} ± {results['cosine_distribution']['cosine_std']:.4f}")

    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    metrics_path = os.path.join(out_dir, "eval_results.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {metrics_path}")

    if args.plot_embeddings:
        sentences = [s for s, _ in SAMPLE_SENTENCES]
        labels = [lbl for _, lbl in SAMPLE_SENTENCES]
        embeddings = get_embeddings_for_visualization(model, tokenizer, sentences, device)

        pca_reduced = reduce_with_pca(embeddings)
        tsne_reduced = reduce_with_tsne(embeddings, perplexity=min(5, len(sentences) - 1))

        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_embedding_projection(pca_reduced, labels, "PCA", cfg.loss.type, plots_dir)
        plot_embedding_projection(tsne_reduced, labels, "tSNE", cfg.loss.type, plots_dir)


if __name__ == "__main__":
    main()
