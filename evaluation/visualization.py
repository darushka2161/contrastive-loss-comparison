import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

sns.set_theme(style="whitegrid", palette="tab10")

COLORS = {"info_nce": "#1f77b4", "triplet": "#ff7f0e", "cosine": "#2ca02c"}
LOSS_LABELS = {"info_nce": "InfoNCE", "triplet": "Triplet", "cosine": "Cosine"}
MODE_LABELS = {
    "nli_only": "NLI only",
    "sts_only": "STS only",
    "nli_plus_sts": "NLI + STS",
}
BACKBONE_COLORS = {
    "MiniLM-L6": "#9467bd",
    "BERT-base": "#1f77b4",
    "RoBERTa-large": "#d62728",
}


def _lr_label(lr: float) -> str:
    return f"lr={lr:.0e}"


def _get_best_lr(lr_logs: Dict, metric_col: str = "spearman_score") -> float:
    """Return the LR with the highest best Spearman from val logs."""
    best_lr = None
    best_val = -1.0
    for lr, logs in lr_logs.items():
        val_df = logs.get("spearman", pd.DataFrame())
        if val_df.empty or metric_col not in val_df.columns:
            continue
        v = val_df[metric_col].max()
        if v > best_val:
            best_val = v
            best_lr = lr
    return best_lr


# ─────────────────────────── Core comparison plots ───────────────────────────

def plot_train_loss(run_logs: Dict[str, Dict[float, Dict]], save_dir: str):
    """Train loss vs steps (best LR per loss function)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for loss_type, lr_logs in run_logs.items():
        best_lr = _get_best_lr(lr_logs)
        if best_lr is None:
            continue
        df = lr_logs[best_lr].get("train", pd.DataFrame())
        if df.empty or "step" not in df.columns:
            continue
        ax.plot(df["step"], df["train_loss"],
                label=LOSS_LABELS.get(loss_type, loss_type),
                color=COLORS.get(loss_type), linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss vs Steps (Best LR per Method)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "train_loss_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_spearman_vs_epochs(run_logs: Dict[str, Dict[float, Dict]], save_dir: str):
    """Spearman validation vs epochs (best LR per loss)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for loss_type, lr_logs in run_logs.items():
        best_lr = _get_best_lr(lr_logs)
        if best_lr is None:
            continue
        df = lr_logs[best_lr].get("spearman", pd.DataFrame())
        if df.empty:
            continue
        ax.plot(df["epoch"], df["spearman_score"], marker="o",
                label=LOSS_LABELS.get(loss_type, loss_type),
                color=COLORS.get(loss_type), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman Correlation (val)")
    ax.set_title("Validation Spearman vs Epochs (Best LR)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "spearman_vs_epochs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_convergence_per_loss(run_logs: Dict[str, Dict[float, Dict]], save_dir: str):
    """Convergence curves at different LRs, one figure per loss function."""
    for loss_type, lr_logs in run_logs.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for lr, logs in sorted(lr_logs.items()):
            label = _lr_label(lr)
            train_df = logs.get("train", pd.DataFrame())
            val_df = logs.get("spearman", pd.DataFrame())
            if not train_df.empty and "step" in train_df.columns:
                axes[0].plot(train_df["step"], train_df["train_loss"], label=label, linewidth=1.5)
            if not val_df.empty:
                axes[1].plot(val_df["epoch"], val_df["spearman_score"],
                             marker="o", label=label, linewidth=1.5)

        axes[0].set_title(f"{LOSS_LABELS.get(loss_type, loss_type)} — Train Loss vs Steps")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=8)

        axes[1].set_title(f"{LOSS_LABELS.get(loss_type, loss_type)} — Spearman(val) vs Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Spearman")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(save_dir, f"convergence_{loss_type}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def plot_spearman_vs_lr(summary_df: pd.DataFrame, save_dir: str):
    """Line chart: best Spearman vs LR for each loss type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for loss_type in summary_df["loss_type"].unique():
        sub = summary_df[summary_df["loss_type"] == loss_type].sort_values("learning_rate")
        col = "best_spearman_test" if "best_spearman_test" in sub.columns else "best_spearman"
        ax.plot(sub["learning_rate"].astype(str), sub[col], marker="o",
                label=LOSS_LABELS.get(loss_type, loss_type),
                color=COLORS.get(loss_type), linewidth=2)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best Spearman (test)")
    ax.set_title("Spearman Correlation vs Learning Rate")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "spearman_vs_lr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_time_vs_quality(summary_df: pd.DataFrame, save_dir: str):
    """Scatter: training time vs best Spearman, coloured by loss type."""
    col = "best_spearman_test" if "best_spearman_test" in summary_df.columns else "best_spearman"
    fig, ax = plt.subplots(figsize=(9, 6))
    for loss_type in summary_df["loss_type"].unique():
        sub = summary_df[summary_df["loss_type"] == loss_type]
        ax.scatter(sub["training_time"] / 60, sub[col],
                   label=LOSS_LABELS.get(loss_type, loss_type),
                   color=COLORS.get(loss_type), s=80, alpha=0.8)
    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Best Spearman (test)")
    ax.set_title("Training Time vs Quality")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "time_vs_quality.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_stability_heatmap(summary_df: pd.DataFrame, save_dir: str):
    """Heatmap: loss × LR coloured by Spearman."""
    col = "best_spearman_test" if "best_spearman_test" in summary_df.columns else "best_spearman"
    pivot = summary_df.pivot_table(index="loss_type", columns="learning_rate", values=col, aggfunc="max")
    pivot.index = [LOSS_LABELS.get(i, i) for i in pivot.index]
    pivot.columns = [f"{c:.0e}" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, linewidths=0.5)
    ax.set_title("Spearman (test) — Loss Type × Learning Rate")
    plt.tight_layout()
    path = os.path.join(save_dir, "stability_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ────────────────── Supervision strategy plots (new) ──────────────────────────

def plot_supervision_comparison(summary_df: pd.DataFrame, save_dir: str):
    """
    Bar chart comparing NLI-only vs STS-only vs NLI+STS strategies.
    Groups by training_mode, shows best Spearman per mode.
    """
    if "training_mode" not in summary_df.columns:
        return

    col = "best_spearman_test" if "best_spearman_test" in summary_df.columns else "best_spearman"
    # Best result per (loss_type, training_mode)
    best = (
        summary_df.groupby(["loss_type", "training_mode"])[col]
        .max()
        .reset_index()
    )
    best["Loss"] = best["loss_type"].map(LOSS_LABELS)
    best["Mode"] = best["training_mode"].map(MODE_LABELS)

    modes = [m for m in ["nli_only", "sts_only", "nli_plus_sts"]
             if m in best["training_mode"].values]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(best["Loss"].unique()))
    width = 0.25
    loss_types = sorted(best["loss_type"].unique())

    for i, mode in enumerate(modes):
        sub = best[best["training_mode"] == mode]
        heights = []
        for lt in loss_types:
            row = sub[sub["loss_type"] == lt]
            heights.append(float(row[col].values[0]) if not row.empty else 0.0)
        bars = ax.bar(x + i * width, heights, width,
                      label=MODE_LABELS.get(mode, mode), alpha=0.85)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels([LOSS_LABELS.get(lt, lt) for lt in loss_types])
    ax.set_ylabel("Best Spearman (test)")
    ax.set_title("Supervision Strategy Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "supervision_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_sts_finetuning_effect(summary_df: pd.DataFrame, save_dir: str):
    """
    Visualise the gain from STS fine-tuning:
    shows Spearman before vs after Stage 2 for nli_plus_sts runs.
    """
    if "training_mode" not in summary_df.columns:
        return
    ft_df = summary_df[
        (summary_df["training_mode"] == "nli_plus_sts") &
        (summary_df["spearman_before_sts_ft"].notna())
    ].copy() if "spearman_before_sts_ft" in summary_df.columns else pd.DataFrame()

    if ft_df.empty:
        return

    col_after = "best_spearman_test" if "best_spearman_test" in ft_df.columns else "best_spearman"
    ft_df["before"] = ft_df["spearman_before_sts_ft"]
    ft_df["after"] = ft_df[col_after]
    ft_df["delta"] = ft_df["after"] - ft_df["before"]
    ft_df["lr_str"] = ft_df["learning_rate"].apply(lambda x: f"{x:.0e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for loss_type in ft_df["loss_type"].unique():
        sub = ft_df[ft_df["loss_type"] == loss_type].sort_values("learning_rate")
        axes[0].plot(sub["lr_str"], sub["before"], marker="o", linestyle="--",
                     label=f"{LOSS_LABELS.get(loss_type)} before FT",
                     color=COLORS.get(loss_type), alpha=0.6)
        axes[0].plot(sub["lr_str"], sub["after"], marker="s",
                     label=f"{LOSS_LABELS.get(loss_type)} after FT",
                     color=COLORS.get(loss_type))

    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Spearman (test)")
    axes[0].set_title("Before vs After STS Fine-tuning")
    axes[0].legend(fontsize=7)

    for loss_type in ft_df["loss_type"].unique():
        sub = ft_df[ft_df["loss_type"] == loss_type].sort_values("learning_rate")
        axes[1].bar(sub["lr_str"], sub["delta"],
                    label=LOSS_LABELS.get(loss_type), alpha=0.75,
                    color=COLORS.get(loss_type))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Δ Spearman (after − before)")
    axes[1].set_title("STS Fine-tuning Delta")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "sts_finetuning_effect.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_backbone_comparison(summary_df: pd.DataFrame, save_dir: str):
    """Bar chart comparing backbone model sizes for best loss/LR configs."""
    if "model_name" not in summary_df.columns:
        return
    from config import BACKBONE_SHORT_NAMES

    col = "best_spearman_test" if "best_spearman_test" in summary_df.columns else "best_spearman"
    df = summary_df.copy()
    df["backbone"] = df["model_name"].map(lambda x: BACKBONE_SHORT_NAMES.get(x, x.split("/")[-1]))

    best = df.groupby(["loss_type", "backbone"])[col].max().reset_index()
    backbones = best["backbone"].unique()
    loss_types = sorted(best["loss_type"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(backbones))
    width = 0.25

    for i, lt in enumerate(loss_types):
        sub = best[best["loss_type"] == lt]
        heights = []
        for bb in backbones:
            row = sub[sub["backbone"] == bb]
            heights.append(float(row[col].values[0]) if not row.empty else 0.0)
        bars = ax.bar(x + i * width, heights, width,
                      label=LOSS_LABELS.get(lt, lt),
                      color=COLORS.get(lt), alpha=0.85)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(loss_types) - 1) / 2)
    ax.set_xticklabels(backbones)
    ax.set_ylabel("Best Spearman (test)")
    ax.set_title("Backbone Model Size Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "backbone_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_grad_norm(run_logs: Dict[str, Dict[float, Dict]], save_dir: str):
    """Gradient norms over training steps (best LR per loss)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False
    for loss_type, lr_logs in run_logs.items():
        best_lr = _get_best_lr(lr_logs)
        if best_lr is None:
            continue
        df = lr_logs[best_lr].get("train", pd.DataFrame())
        if df.empty or "grad_norm" not in df.columns:
            continue
        ax.plot(df["step"], df["grad_norm"],
                label=LOSS_LABELS.get(loss_type, loss_type),
                color=COLORS.get(loss_type), linewidth=1.5, alpha=0.8)
        any_data = True

    if not any_data:
        plt.close()
        return
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm (clipped)")
    ax.set_title("Gradient Norms During Training (Best LR)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "grad_norm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_embedding_projection(reduced: np.ndarray, labels: List[str],
                               method: str, loss_type: str, save_dir: str):
    """2D projection of embeddings (PCA or t-SNE)."""
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(reduced[idx, 0], reduced[idx, 1], label=lbl,
                   color=color_map[lbl], alpha=0.7, s=30)
    ax.set_title(f"{method} of Embeddings — {LOSS_LABELS.get(loss_type, loss_type)}")
    ax.legend(markerscale=2)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{method.lower()}_{loss_type}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
