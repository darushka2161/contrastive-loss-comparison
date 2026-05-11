#!/usr/bin/env python3
"""
Run the full experiment grid.

Stage 1 — Loss function comparison (all loss types × all LRs):
  - NLI-only training for InfoNCE, Triplet, Cosine
  - STS-only training for Cosine
  - NLI+STS two-stage training for Cosine

Stage 2 — Backbone comparison (best loss + best LR × multiple backbones).
"""
import argparse
import json
import os

import pandas as pd

from config import load_config, get_run_dir, BACKBONE_SHORT_NAMES
from training.trainer import train_one_run
from training.train_utils import set_seed

# Stage 1: supervision strategy sweep
STAGE1_CONFIGS = {
    "info_nce": "experiments/info_nce_nli.yaml",
    "triplet":  "experiments/triplet_nli.yaml",
    "cosine":   "experiments/cosine_nli.yaml",
    "cosine_sts":          "experiments/cosine_sts.yaml",
    "cosine_nli_plus_sts": "experiments/cosine_nli_plus_sts.yaml",
}

# Legacy single-mode configs (kept for backward compatibility)
LEGACY_CONFIGS = {
    "info_nce": "experiments/info_nce.yaml",
    "triplet":  "experiments/triplet.yaml",
    "cosine":   "experiments/cosine.yaml",
}


def _result_path(run_dir: str) -> str:
    return os.path.join(run_dir, "metrics", "result.json")


def _load_existing(run_dir: str):
    p = _result_path(run_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def run_stage1(args) -> list:
    """Run all loss × LR × supervision mode experiments."""
    configs_to_run = {k: v for k, v in STAGE1_CONFIGS.items()
                      if k in args.experiments}
    all_results = []

    for exp_name, config_path in configs_to_run.items():
        if not os.path.exists(config_path):
            print(f"  [SKIP] Config not found: {config_path}")
            continue

        cfg = load_config(config_path)
        cfg.output_dir = args.output_dir
        if args.max_train_samples:
            cfg.training.max_train_samples = args.max_train_samples

        print(f"\n{'='*65}")
        print(f"Experiment: {exp_name.upper()} | mode={cfg.training_mode}")
        print(f"{'='*65}")

        for lr in cfg.learning_rates:
            run_dir = get_run_dir(args.output_dir, cfg.loss.type, lr,
                                  cfg.training_mode, cfg.model_name)

            if args.skip_existing and _load_existing(run_dir):
                print(f"  [SKIP] {exp_name} | lr={lr}")
                all_results.append(_load_existing(run_dir))
                continue

            print(f"\n  Running: {exp_name} | lr={lr}")
            result = train_one_run(cfg, lr=lr, run_dir=run_dir)

            with open(_result_path(run_dir), "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

    return all_results


def run_stage2(args, stage1_df: pd.DataFrame) -> list:
    """Backbone size comparison using best loss+LR from Stage 1."""
    if stage1_df.empty:
        print("No Stage 1 results for Stage 2. Skipping.")
        return []

    col = "best_spearman_test" if "best_spearman_test" in stage1_df.columns else "best_spearman"
    # Best NLI-only config (most like a clean loss comparison)
    nli_df = stage1_df[stage1_df.get("training_mode", "nli_only") == "nli_only"] \
        if "training_mode" in stage1_df.columns else stage1_df

    if nli_df.empty:
        nli_df = stage1_df

    best_row = nli_df.loc[nli_df[col].idxmax()]
    best_loss = best_row["loss_type"]
    best_lr = float(best_row["learning_rate"])

    print(f"\n{'='*65}")
    print(f"Stage 2: Backbone comparison")
    print(f"  Best config from Stage 1: {best_loss} | lr={best_lr:.0e}")
    print(f"{'='*65}")

    config_map = {
        "info_nce": "experiments/info_nce_nli.yaml",
        "triplet":  "experiments/triplet_nli.yaml",
        "cosine":   "experiments/cosine_nli.yaml",
    }
    cfg_path = config_map.get(best_loss, "experiments/cosine_nli.yaml")
    cfg = load_config(cfg_path)
    cfg.output_dir = args.output_dir
    if args.max_train_samples:
        cfg.training.max_train_samples = args.max_train_samples

    from config import ALL_BACKBONES
    all_results = []
    for backbone in ALL_BACKBONES:
        run_dir = get_run_dir(args.output_dir, cfg.loss.type, best_lr,
                              cfg.training_mode, backbone)
        if args.skip_existing and _load_existing(run_dir):
            print(f"  [SKIP] {BACKBONE_SHORT_NAMES.get(backbone, backbone)}")
            all_results.append(_load_existing(run_dir))
            continue

        print(f"\n  Backbone: {BACKBONE_SHORT_NAMES.get(backbone, backbone)}")
        result = train_one_run(cfg, lr=best_lr, run_dir=run_dir,
                               model_name_override=backbone)
        with open(_result_path(run_dir), "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

    return all_results


def collect_all_results(output_dir: str) -> pd.DataFrame:
    rows = []
    for root, dirs, files in os.walk(output_dir):
        if "result.json" in files:
            with open(os.path.join(root, "result.json")) as f:
                rows.append(json.load(f))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_plots(output_dir: str, df: pd.DataFrame):
    from evaluation.visualization import (
        plot_spearman_vs_lr, plot_time_vs_quality, plot_stability_heatmap,
        plot_convergence_per_loss, plot_spearman_vs_epochs, plot_train_loss,
        plot_supervision_comparison, plot_sts_finetuning_effect,
        plot_backbone_comparison, plot_grad_norm,
    )

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Summary plots
    plot_spearman_vs_lr(df, plots_dir)
    plot_time_vs_quality(df, plots_dir)
    plot_stability_heatmap(df, plots_dir)
    plot_supervision_comparison(df, plots_dir)
    plot_sts_finetuning_effect(df, plots_dir)
    plot_backbone_comparison(df, plots_dir)

    # Per-run logs
    run_logs = _load_run_logs(output_dir)
    if run_logs:
        plot_train_loss(run_logs, plots_dir)
        plot_spearman_vs_epochs(run_logs, plots_dir)
        plot_convergence_per_loss(run_logs, plots_dir)
        plot_grad_norm(run_logs, plots_dir)


def _load_run_logs(output_dir: str) -> dict:
    run_logs = {}
    for root, dirs, files in os.walk(output_dir):
        if "result.json" not in files:
            continue
        with open(os.path.join(root, "result.json")) as f:
            r = json.load(f)
        loss_type = r.get("loss_type", "unknown")
        lr_val = r.get("learning_rate", 0.0)

        train_log = os.path.join(root, "..", "logs", "train_log.csv")
        val_log   = os.path.join(root, "..", "logs", "val_log.csv")
        train_df = pd.read_csv(train_log) if os.path.exists(train_log) else pd.DataFrame()
        val_df   = pd.read_csv(val_log)   if os.path.exists(val_log)   else pd.DataFrame()

        run_logs.setdefault(loss_type, {})[lr_val] = {
            "train": train_df, "spearman": val_df,
        }
    return {lt: lrs for lt, lrs in run_logs.items() if lrs}


def print_summary(df: pd.DataFrame):
    if df.empty:
        print("No results to display.")
        return

    col = "best_spearman_test" if "best_spearman_test" in df.columns else "best_spearman"
    print("\n" + "=" * 70)
    print("SUMMARY — Best Result per Loss × Mode")
    print("=" * 70)

    group_cols = ["loss_type"]
    if "training_mode" in df.columns:
        group_cols.append("training_mode")

    best = (
        df.sort_values(col, ascending=False)
        .groupby(group_cols)
        .first()
        .reset_index()
    )

    stability = df.groupby(group_cols)[col].agg(mean="mean", std="std").reset_index()
    merged = best.merge(stability, on=group_cols)

    display = merged[group_cols + ["learning_rate", col, "mean", "std", "training_time"]]
    print(display.to_string(index=False))
    print("=" * 70)

    best_overall = df.loc[df[col].idxmax()]
    print(f"\nOverall best: {best_overall['loss_type']} "
          f"| lr={best_overall['learning_rate']:.0e} "
          f"| Spearman={best_overall[col]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--stage2", action="store_true", help="Also run backbone comparison")
    parser.add_argument("--experiments", nargs="+",
                        default=list(STAGE1_CONFIGS.keys()),
                        help="Which experiment keys to run from STAGE1_CONFIGS")
    args = parser.parse_args()

    # Stage 1
    stage1_results = run_stage1(args)

    # Collect all results (including pre-existing)
    df = collect_all_results(args.output_dir)

    # Stage 2 (optional)
    if args.stage2 and not df.empty:
        stage2_results = run_stage2(args, df)
        df = collect_all_results(args.output_dir)  # reload

    print_summary(df)

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)
    print(f"\nSaved: {os.path.join(args.output_dir, 'all_results.csv')}")

    print("\nBuilding plots...")
    build_plots(args.output_dir, df)

    print("\nDone!")


if __name__ == "__main__":
    main()
