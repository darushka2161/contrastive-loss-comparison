#!/usr/bin/env python3
"""
Load all experiment results and produce:
  1. Comparison table: best Spearman per loss × mode × LR
  2. Supervision strategy comparison table
  3. Backbone comparison table (if Stage 2 was run)
  4. Stability analysis (sensitivity to LR choice)
  5. STS fine-tuning effect
  6. All comparative plots
"""
import argparse
import json
import os

import pandas as pd
import numpy as np

from config import BACKBONE_SHORT_NAMES
from evaluation.visualization import (
    plot_spearman_vs_lr, plot_time_vs_quality, plot_stability_heatmap,
    plot_convergence_per_loss, plot_spearman_vs_epochs, plot_train_loss,
    plot_supervision_comparison, plot_sts_finetuning_effect,
    plot_backbone_comparison, plot_grad_norm,
)

LOSS_LABELS = {"info_nce": "InfoNCE", "triplet": "Triplet", "cosine": "Cosine"}
MODE_LABELS = {
    "nli_only": "NLI only",
    "sts_only": "STS only",
    "nli_plus_sts": "NLI + STS",
}


def load_all_results(output_dir: str) -> pd.DataFrame:
    rows = []
    for root, dirs, files in os.walk(output_dir):
        if "result.json" in files:
            try:
                with open(os.path.join(root, "result.json")) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_run_logs(output_dir: str) -> dict:
    run_logs = {}
    for root, dirs, files in os.walk(output_dir):
        if "result.json" not in files:
            continue
        try:
            with open(os.path.join(root, "result.json")) as f:
                r = json.load(f)
        except Exception:
            continue
        loss_type = r.get("loss_type", "unknown")
        lr_val = r.get("learning_rate", 0.0)
        logs_dir = os.path.join(root, "..", "logs")
        train_log = os.path.join(logs_dir, "train_log.csv")
        val_log   = os.path.join(logs_dir, "val_log.csv")
        train_df = pd.read_csv(train_log) if os.path.exists(train_log) else pd.DataFrame()
        val_df   = pd.read_csv(val_log)   if os.path.exists(val_log)   else pd.DataFrame()
        run_logs.setdefault(loss_type, {})[lr_val] = {"train": train_df, "spearman": val_df}
    return {lt: lrs for lt, lrs in run_logs.items() if lrs}


def _spearman_col(df: pd.DataFrame) -> str:
    return "best_spearman_test" if "best_spearman_test" in df.columns else "best_spearman"


def print_main_comparison(df: pd.DataFrame):
    col = _spearman_col(df)
    print("\n" + "=" * 75)
    print("TABLE 1 — Best Result per Loss Function")
    print("=" * 75)

    group_cols = ["loss_type"]
    if "training_mode" in df.columns:
        nli_df = df[df["training_mode"] == "nli_only"]
    else:
        nli_df = df

    best = (
        nli_df.sort_values(col, ascending=False)
        .groupby("loss_type")
        .first()
        .reset_index()
    )
    stability = nli_df.groupby("loss_type")[col].agg(mean="mean", std="std").reset_index()
    merged = best.merge(stability, on="loss_type")
    merged["Loss"] = merged["loss_type"].map(LOSS_LABELS)
    merged["Best LR"] = merged["learning_rate"].apply(lambda x: f"{x:.0e}")
    merged["Best Spearman"] = merged[col].apply(lambda x: f"{x:.4f}")
    merged["Mean ± Std"] = merged.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
    )
    merged["Time (s)"] = merged["training_time"].apply(lambda x: f"{x:.0f}s")
    merged["Diverged"] = merged.get("diverged", False)

    cols = ["Loss", "Best LR", "Best Spearman", "Mean ± Std", "Time (s)"]
    if "diverged" in merged.columns:
        cols.append("Diverged")
    print(merged[cols].to_string(index=False))
    print("=" * 75)

    best_method = merged.loc[merged[col].idxmax(), "Loss"]
    stable_method = merged.loc[merged["std"].idxmin(), "Loss"]
    fastest = merged.loc[merged["training_time"].idxmin(), "Loss"]
    print(f"\n  Best performance:    {best_method}")
    print(f"  Most stable (LR):    {stable_method}")
    print(f"  Fastest training:    {fastest}")


def print_supervision_comparison(df: pd.DataFrame):
    if "training_mode" not in df.columns:
        return
    col = _spearman_col(df)
    print("\n" + "=" * 75)
    print("TABLE 2 — Supervision Strategy Comparison")
    print("=" * 75)
    best = (
        df.sort_values(col, ascending=False)
        .groupby(["training_mode", "loss_type"])
        .first()
        .reset_index()
    )
    best["Loss"] = best["loss_type"].map(LOSS_LABELS)
    best["Mode"] = best["training_mode"].map(MODE_LABELS)
    best["Best Spearman"] = best[col].apply(lambda x: f"{x:.4f}")
    best["Best LR"] = best["learning_rate"].apply(lambda x: f"{x:.0e}")
    print(best[["Mode", "Loss", "Best LR", "Best Spearman"]].to_string(index=False))
    print("=" * 75)

    # Is NLI+STS better than NLI-only?
    nli_scores = best[best["training_mode"] == "nli_only"][col].max() \
        if "nli_only" in best["training_mode"].values else None
    ft_scores = best[best["training_mode"] == "nli_plus_sts"][col].max() \
        if "nli_plus_sts" in best["training_mode"].values else None
    if nli_scores and ft_scores:
        delta = ft_scores - nli_scores
        sign = "↑" if delta > 0 else "↓"
        print(f"\n  STS fine-tuning effect: {sign} {abs(delta):.4f} Spearman vs NLI-only")

    # STS-only vs NLI-only
    sts_scores = best[best["training_mode"] == "sts_only"][col].max() \
        if "sts_only" in best["training_mode"].values else None
    if nli_scores and sts_scores:
        delta = sts_scores - nli_scores
        sign = "↑" if delta > 0 else "↓"
        print(f"  STS-only vs NLI-only:   {sign} {abs(delta):.4f} Spearman")


def print_backbone_comparison(df: pd.DataFrame):
    if "model_name" not in df.columns or df["model_name"].nunique() <= 1:
        return
    col = _spearman_col(df)
    print("\n" + "=" * 75)
    print("TABLE 3 — Backbone Size Comparison")
    print("=" * 75)
    df2 = df.copy()
    df2["backbone"] = df2["model_name"].map(
        lambda x: BACKBONE_SHORT_NAMES.get(x, x.split("/")[-1])
    )
    best = (
        df2.sort_values(col, ascending=False)
        .groupby(["backbone", "loss_type"])
        .first()
        .reset_index()
    )
    best["Loss"] = best["loss_type"].map(LOSS_LABELS)
    best["Best Spearman"] = best[col].apply(lambda x: f"{x:.4f}")
    best["Best LR"] = best["learning_rate"].apply(lambda x: f"{x:.0e}")
    print(best[["backbone", "Loss", "Best LR", "Best Spearman"]].to_string(index=False))
    print("=" * 75)


def print_sensitivity_analysis(df: pd.DataFrame):
    col = _spearman_col(df)
    nli_df = df[df["training_mode"] == "nli_only"] \
        if "training_mode" in df.columns else df
    if nli_df.empty:
        return

    print("\n" + "=" * 75)
    print("SENSITIVITY ANALYSIS — How sensitive each loss is to LR choice")
    print("=" * 75)
    sens = nli_df.groupby("loss_type")[col].agg(
        best="max", worst="min", mean="mean", std="std", range=lambda x: x.max() - x.min()
    ).reset_index()
    sens["Loss"] = sens["loss_type"].map(LOSS_LABELS)
    sens["Sensitivity (std)"] = sens["std"].apply(lambda x: f"{x:.4f}")
    sens["Range (max-min)"] = sens["range"].apply(lambda x: f"{x:.4f}")
    sens["Best"] = sens["best"].apply(lambda x: f"{x:.4f}")
    sens["Worst"] = sens["worst"].apply(lambda x: f"{x:.4f}")
    print(sens[["Loss", "Best", "Worst", "Sensitivity (std)", "Range (max-min)"]].to_string(index=False))
    print("=" * 75)
    most_sensitive = sens.loc[sens["std"].idxmax(), "Loss"]
    most_robust = sens.loc[sens["std"].idxmin(), "Loss"]
    print(f"  Most sensitive to LR: {most_sensitive}")
    print(f"  Most robust to LR:    {most_robust}")


def print_divergence_report(df: pd.DataFrame):
    if "diverged" not in df.columns:
        return
    div_df = df[df["diverged"] == True]
    if div_df.empty:
        print("\n  No divergences detected across all runs.")
        return
    print("\n" + "=" * 75)
    print("DIVERGENCE REPORT — Runs with NaN/Inf loss")
    print("=" * 75)
    cols = ["loss_type", "learning_rate"]
    if "training_mode" in div_df.columns:
        cols.append("training_mode")
    print(div_df[cols].to_string(index=False))
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--plots_dir", default=None)
    args = parser.parse_args()

    plots_dir = args.plots_dir or os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = load_all_results(args.output_dir)
    if df.empty:
        print("No results found. Run run_all_experiments.py first.")
        return

    # Print all analysis tables
    print_main_comparison(df)
    print_supervision_comparison(df)
    print_backbone_comparison(df)
    print_sensitivity_analysis(df)
    print_divergence_report(df)

    # Save CSVs
    df.to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)

    # Generate all plots
    print("\nGenerating plots...")
    run_logs = load_run_logs(args.output_dir)

    plot_spearman_vs_lr(df, plots_dir)
    plot_time_vs_quality(df, plots_dir)
    plot_stability_heatmap(df, plots_dir)
    plot_supervision_comparison(df, plots_dir)
    plot_sts_finetuning_effect(df, plots_dir)
    plot_backbone_comparison(df, plots_dir)

    if run_logs:
        plot_train_loss(run_logs, plots_dir)
        plot_spearman_vs_epochs(run_logs, plots_dir)
        plot_convergence_per_loss(run_logs, plots_dir)
        plot_grad_norm(run_logs, plots_dir)

    print(f"\nAll results and plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
