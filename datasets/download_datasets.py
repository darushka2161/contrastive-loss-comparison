#!/usr/bin/env python3
"""Download and preprocess all required datasets."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.prepare_nli import prepare_and_save
from datasets.prepare_sts import prepare_sts_and_save


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit NLI training samples (None = all)")
    args = parser.parse_args()

    print("=" * 60)
    print("Downloading and preparing NLI datasets (SNLI + MNLI)...")
    print("=" * 60)
    prepare_and_save(
        processed_dir=args.processed_dir,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("Downloading and preparing STS Benchmark (train/val/test)...")
    print("  NOTE: STS test is reserved for final evaluation only.")
    print("        STS val  is used for validation during training.")
    print("        STS train is used for Stage 2 fine-tuning.")
    print("=" * 60)
    prepare_sts_and_save(
        processed_dir=args.processed_dir,
        cache_dir=args.cache_dir,
    )

    print("\nAll datasets prepared successfully!")


if __name__ == "__main__":
    main()
