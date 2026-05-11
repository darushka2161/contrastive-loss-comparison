import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, List

from config import (
    ExperimentConfig, get_run_dir,
    TRAINING_MODE_NLI_ONLY, TRAINING_MODE_STS_ONLY, TRAINING_MODE_NLI_PLUS_STS,
)
from models.sentence_encoder import SentenceEncoder, get_tokenizer
from losses.info_nce import InfoNCELoss
from losses.triplet import TripletLoss
from losses.cosine_loss import CosineSimilarityLoss
from datasets.dataset_utils import (
    NLIPairDataset, NLITripletDataset, NLICosineDataset,
    collate_pair, collate_triplet, collate_cosine,
)
from datasets.prepare_nli import load_processed
from datasets.prepare_sts import load_sts_processed, load_sts_cosine_tuples
from evaluation.sts_evaluator import evaluate_on_sts
from training.train_utils import set_seed, get_device, CSVLogger, EarlyStopping


def build_loss(cfg: ExperimentConfig) -> nn.Module:
    t = cfg.loss.type
    if t == "info_nce":
        return InfoNCELoss(temperature=cfg.loss.temperature)
    elif t == "triplet":
        return TripletLoss(margin=cfg.loss.margin)
    elif t == "cosine":
        return CosineSimilarityLoss()
    raise ValueError(f"Unknown loss type: {t}")


def _make_nli_loader(cfg: ExperimentConfig, tokenizer, max_samples: Optional[int]) -> DataLoader:
    """Build DataLoader for NLI training data."""
    loss_type = cfg.loss.type
    data = load_processed("data/processed", loss_type)
    if max_samples:
        data = data[:max_samples]

    max_length = 128
    if loss_type == "info_nce":
        dataset = NLIPairDataset(data)
        collate_fn = partial(collate_pair, tokenizer=tokenizer, max_length=max_length)
    elif loss_type == "triplet":
        dataset = NLITripletDataset(data)
        collate_fn = partial(collate_triplet, tokenizer=tokenizer, max_length=max_length)
    else:
        dataset = NLICosineDataset(data)
        collate_fn = partial(collate_cosine, tokenizer=tokenizer, max_length=max_length)

    return DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True,
                      num_workers=0, collate_fn=collate_fn, drop_last=True)


def _make_sts_loader(cfg: ExperimentConfig, tokenizer) -> DataLoader:
    """Build DataLoader for STS train split (always CosineSimilarityLoss format)."""
    data = load_sts_cosine_tuples("data/processed")
    dataset = NLICosineDataset(data)
    collate_fn = partial(collate_cosine, tokenizer=tokenizer, max_length=128)
    return DataLoader(dataset, batch_size=min(cfg.training.batch_size, 32), shuffle=True,
                      num_workers=0, collate_fn=collate_fn, drop_last=False)


def _forward_batch(model, loss_fn, batch, loss_type: str, device, use_amp: bool):
    if loss_type == "info_nce":
        enc_a, enc_p = batch
        enc_a = {k: v.to(device) for k, v in enc_a.items()}
        enc_p = {k: v.to(device) for k, v in enc_p.items()}
        with autocast(enabled=use_amp):
            loss = loss_fn(model.encode_batch(enc_a), model.encode_batch(enc_p))

    elif loss_type == "triplet":
        enc_a, enc_p, enc_n = batch
        enc_a = {k: v.to(device) for k, v in enc_a.items()}
        enc_p = {k: v.to(device) for k, v in enc_p.items()}
        enc_n = {k: v.to(device) for k, v in enc_n.items()}
        with autocast(enabled=use_amp):
            loss = loss_fn(model.encode_batch(enc_a), model.encode_batch(enc_p),
                           model.encode_batch(enc_n))

    else:  # cosine
        enc1, enc2, scores = batch
        enc1 = {k: v.to(device) for k, v in enc1.items()}
        enc2 = {k: v.to(device) for k, v in enc2.items()}
        scores = scores.to(device)
        with autocast(enabled=use_amp):
            loss = loss_fn(model.encode_batch(enc1), model.encode_batch(enc2), scores)

    return loss


def _train_epoch(model, loss_fn, loader, optimizer, scheduler, scaler, cfg, device,
                 use_amp: bool, epoch: int, global_step: int,
                 train_logger: CSVLogger, loss_type_override: str = None) -> tuple:
    """Run one training epoch. Returns (updated_global_step, epoch_avg_loss, diverged)."""
    model.train()
    loss_type = loss_type_override or cfg.loss.type
    acc_steps = cfg.training.gradient_accumulation_steps
    optimizer.zero_grad()
    running_loss = 0.0
    diverged = False

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar, 1):
        loss = _forward_batch(model, loss_fn, batch, loss_type, device, use_amp)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  [WARNING] NaN/Inf loss at step {step}, epoch {epoch}. Stopping.")
            diverged = True
            break

        (loss / acc_steps).backward() if not use_amp else \
            scaler.scale(loss / acc_steps).backward()

        running_loss += loss.item()

        if step % acc_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_loss = running_loss / step
            train_logger.log({
                "epoch": epoch,
                "step": global_step,
                "train_loss": round(avg_loss, 6),
                "learning_rate": round(scheduler.get_last_lr()[0], 8),
                "grad_norm": round(float(grad_norm), 4),
            })

        pbar.set_postfix({"loss": f"{running_loss / step:.4f}"})

    return global_step, running_loss / max(len(loader), 1), diverged


def train_one_run(cfg: ExperimentConfig, lr: float, run_dir: str,
                  model_name_override: str = None) -> Dict:
    """
    Train a model with the given config and LR.

    Supports three training_mode values:
    - nli_only:     Train on NLI corpus only.
    - sts_only:     Train on STS train split with CosineSimilarityLoss.
    - nli_plus_sts: Stage 1 on NLI, Stage 2 fine-tuning on STS train.
    """
    set_seed(cfg.seed)
    device = get_device()
    model_name = model_name_override or cfg.model_name

    print(f"\n[{cfg.loss.type} | {cfg.training_mode} | lr={lr} | {model_name}] Device: {device}")

    tokenizer = get_tokenizer(model_name)
    model = SentenceEncoder(model_name, cfg.pooling).to(device)

    # Evaluation uses STS val (dev) during training; test is held out for final eval
    sts_val = load_sts_processed("data/processed", "val")

    use_amp = cfg.training.fp16 and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    early_stopper = EarlyStopping(patience=cfg.training.early_stopping_patience)

    train_logger = CSVLogger(
        os.path.join(run_dir, "logs", "train_log.csv"),
        ["epoch", "step", "train_loss", "learning_rate", "grad_norm"],
    )
    val_logger = CSVLogger(
        os.path.join(run_dir, "logs", "val_log.csv"),
        ["epoch", "spearman_score", "stage", "epoch_time_s"],
    )

    best_spearman = -1.0
    global_step = 0
    start_time = time.time()
    diverged = False
    epochs_run = 0

    # ── Stage 1: NLI training (or STS-only) ──────────────────────────────────
    if cfg.training_mode in (TRAINING_MODE_NLI_ONLY, TRAINING_MODE_NLI_PLUS_STS):
        stage_loss_fn = build_loss(cfg)
        train_loader = _make_nli_loader(cfg, tokenizer, cfg.training.max_train_samples)
        total_steps = (cfg.training.epochs * len(train_loader)
                       // cfg.training.gradient_accumulation_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=total_steps,
        )

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()
            global_step, _, diverged = _train_epoch(
                model, stage_loss_fn, train_loader, optimizer, scheduler,
                scaler, cfg, device, use_amp, epoch, global_step, train_logger,
            )
            epoch_time = time.time() - t0
            epochs_run = epoch

            spearman = evaluate_on_sts(model, tokenizer, sts_val, device)
            print(f"  [NLI S1] Epoch {epoch} — Spearman(val): {spearman:.4f} "
                  f"| time: {epoch_time:.1f}s")
            val_logger.log({"epoch": epoch, "spearman_score": round(spearman, 6),
                            "stage": "nli", "epoch_time_s": round(epoch_time, 1)})

            if spearman > best_spearman:
                best_spearman = spearman
                torch.save(model.state_dict(),
                           os.path.join(run_dir, "checkpoints", "best_model.pt"))

            if diverged or early_stopper(spearman):
                if not diverged:
                    print(f"  Early stopping at epoch {epoch}")
                break

    elif cfg.training_mode == TRAINING_MODE_STS_ONLY:
        # STS-only always uses CosineSimilarityLoss regardless of cfg.loss.type
        stage_loss_fn = CosineSimilarityLoss()
        train_loader = _make_sts_loader(cfg, tokenizer)
        total_steps = (cfg.training.epochs * len(train_loader)
                       // cfg.training.gradient_accumulation_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=total_steps,
        )

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()
            global_step, _, diverged = _train_epoch(
                model, stage_loss_fn, train_loader, optimizer, scheduler,
                scaler, cfg, device, use_amp, epoch, global_step, train_logger,
                loss_type_override="cosine",
            )
            epoch_time = time.time() - t0
            epochs_run = epoch

            spearman = evaluate_on_sts(model, tokenizer, sts_val, device)
            print(f"  [STS]  Epoch {epoch} — Spearman(val): {spearman:.4f} "
                  f"| time: {epoch_time:.1f}s")
            val_logger.log({"epoch": epoch, "spearman_score": round(spearman, 6),
                            "stage": "sts", "epoch_time_s": round(epoch_time, 1)})

            if spearman > best_spearman:
                best_spearman = spearman
                torch.save(model.state_dict(),
                           os.path.join(run_dir, "checkpoints", "best_model.pt"))

            if diverged or early_stopper(spearman):
                break

    # ── Stage 2: STS fine-tuning (only for nli_plus_sts) ─────────────────────
    if cfg.training_mode == TRAINING_MODE_NLI_PLUS_STS and not diverged:
        print(f"  [Stage 2] STS fine-tuning for {cfg.training.sts_finetune_epochs} epoch(s)...")
        spearman_before_ft = best_spearman

        sts_loss_fn = CosineSimilarityLoss()
        sts_loader = _make_sts_loader(cfg, tokenizer)
        ft_steps = (cfg.training.sts_finetune_epochs * len(sts_loader)
                    // cfg.training.gradient_accumulation_steps)
        ft_lr = lr * 0.5  # lower LR for fine-tuning
        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-2)
        ft_scheduler = get_linear_schedule_with_warmup(
            ft_optimizer, num_warmup_steps=10, num_training_steps=max(ft_steps, 1),
        )
        ft_scaler = GradScaler() if use_amp else None

        for ft_epoch in range(1, cfg.training.sts_finetune_epochs + 1):
            t0 = time.time()
            global_step, _, _ = _train_epoch(
                model, sts_loss_fn, sts_loader, ft_optimizer, ft_scheduler,
                ft_scaler, cfg, device, use_amp,
                epoch=epochs_run + ft_epoch,
                global_step=global_step,
                train_logger=train_logger,
                loss_type_override="cosine",
            )
            epoch_time = time.time() - t0
            spearman = evaluate_on_sts(model, tokenizer, sts_val, device)
            print(f"  [STS FT] Epoch {ft_epoch} — Spearman(val): {spearman:.4f} "
                  f"| time: {epoch_time:.1f}s")
            val_logger.log({
                "epoch": epochs_run + ft_epoch,
                "spearman_score": round(spearman, 6),
                "stage": "sts_finetune",
                "epoch_time_s": round(epoch_time, 1),
            })

            if spearman > best_spearman:
                best_spearman = spearman
                torch.save(model.state_dict(),
                           os.path.join(run_dir, "checkpoints", "best_model_ft.pt"))
    else:
        spearman_before_ft = None

    # ── Final evaluation on STS test ─────────────────────────────────────────
    sts_test = load_sts_processed("data/processed", "test")
    ckpt = os.path.join(run_dir, "checkpoints", "best_model_ft.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "checkpoints", "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    final_spearman = evaluate_on_sts(model, tokenizer, sts_test, device)
    print(f"  Final Spearman (test): {final_spearman:.4f}")

    training_time = time.time() - start_time
    return {
        "loss_type": cfg.loss.type,
        "training_mode": cfg.training_mode,
        "model_name": model_name,
        "learning_rate": lr,
        "best_spearman_val": best_spearman,
        "best_spearman_test": final_spearman,
        "spearman_before_sts_ft": spearman_before_ft,
        "training_time": round(training_time, 2),
        "epochs_run": epochs_run,
        "diverged": diverged,
    }
