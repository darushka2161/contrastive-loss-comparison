import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, List

# Training modes
TRAINING_MODE_NLI_ONLY = "nli_only"
TRAINING_MODE_STS_ONLY = "sts_only"
TRAINING_MODE_NLI_PLUS_STS = "nli_plus_sts"

BACKBONE_SMALL = "sentence-transformers/all-MiniLM-L6-v2"
BACKBONE_MEDIUM = "bert-base-uncased"
BACKBONE_LARGE = "roberta-large"

ALL_BACKBONES = [BACKBONE_SMALL, BACKBONE_MEDIUM, BACKBONE_LARGE]

BACKBONE_SHORT_NAMES = {
    BACKBONE_SMALL: "MiniLM-L6",
    BACKBONE_MEDIUM: "BERT-base",
    BACKBONE_LARGE: "RoBERTa-large",
}


@dataclass
class LossConfig:
    type: str = "info_nce"
    temperature: float = 0.05
    margin: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 64
    lr: float = 2e-5
    epochs: int = 3
    sts_finetune_epochs: int = 1  # epochs for Stage 2 STS fine-tuning
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    eval_steps: int = 500
    save_steps: int = 1000
    max_train_samples: Optional[int] = None
    num_trainable_layers: int = 2  # 0 = train all; N = freeze all except last N transformer blocks


@dataclass
class ExperimentConfig:
    model_name: str = "bert-base-uncased"
    pooling: str = "mean"
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs"
    experiment_name: str = "experiment"
    training_mode: str = TRAINING_MODE_NLI_ONLY  # nli_only | sts_only | nli_plus_sts
    seed: int = 42
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 2e-5, 3e-5, 5e-5])
    backbone_models: List[str] = field(default_factory=lambda: [BACKBONE_MEDIUM])


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    loss_raw = raw.get("loss", {})
    train_raw = raw.get("training", {})

    # Filter only known fields to avoid TypeError
    loss_fields = {f.name for f in LossConfig.__dataclass_fields__.values()}
    train_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}

    loss_cfg = LossConfig(**{k: v for k, v in loss_raw.items() if k in loss_fields})
    train_cfg = TrainingConfig(**{k: v for k, v in train_raw.items() if k in train_fields})

    cfg = ExperimentConfig(
        model_name=raw.get("model_name", "bert-base-uncased"),
        pooling=raw.get("pooling", "mean"),
        loss=loss_cfg,
        training=train_cfg,
        output_dir=raw.get("output_dir", "outputs"),
        experiment_name=raw.get("experiment_name", "experiment"),
        training_mode=raw.get("training_mode", TRAINING_MODE_NLI_ONLY),
        seed=raw.get("seed", 42),
        learning_rates=raw.get("learning_rates", [1e-5, 2e-5, 3e-5, 5e-5]),
        backbone_models=raw.get("backbone_models", [BACKBONE_MEDIUM]),
    )
    return cfg


def get_run_dir(output_dir: str, loss_type: str, lr: float,
                training_mode: str = "nli_only", model_name: str = "") -> str:
    lr_str = f"lr_{str(lr).replace('-', 'neg').replace('.', '_')}"
    mode_str = training_mode
    if model_name:
        short = BACKBONE_SHORT_NAMES.get(model_name, model_name.split("/")[-1])
        parts = [loss_type, mode_str, short, lr_str]
    else:
        parts = [loss_type, mode_str, lr_str]
    run_dir = os.path.join(output_dir, *parts)
    for sub in ("checkpoints", "logs", "metrics", "plots"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir
