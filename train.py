#!/usr/bin/env python3
"""Train a single model run with one loss function, one LR, one training mode."""
import argparse
import json
import os

from config import load_config, get_run_dir
from training.trainer import train_one_run


def main():
    parser = argparse.ArgumentParser(description="Train sentence encoder with contrastive loss")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None, help="Override model_name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_train_samples:
        cfg.training.max_train_samples = args.max_train_samples

    model_name = args.backbone or cfg.model_name
    lr = args.lr if args.lr else cfg.training.lr
    run_dir = get_run_dir(cfg.output_dir, cfg.loss.type, lr, cfg.training_mode, model_name)

    print(f"Starting: {cfg.loss.type} | mode={cfg.training_mode} | lr={lr} | {model_name}")
    print(f"  Output: {run_dir}")

    result = train_one_run(cfg, lr=lr, run_dir=run_dir, model_name_override=model_name)

    result_path = os.path.join(run_dir, "metrics", "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone!")
    print(f"  Best Spearman (val):  {result['best_spearman_val']:.4f}")
    print(f"  Best Spearman (test): {result['best_spearman_test']:.4f}")
    print(f"  Training time: {result['training_time']:.1f}s")
    print(f"  Results: {result_path}")


if __name__ == "__main__":
    main()
