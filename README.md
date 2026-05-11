# Contrastive Loss Comparison for Sentence Embeddings

**«Влияние выбора контрастивной функции потерь на качество обучения предложенческих эмбеддингов»**

---

## Обоснование выбора функций потерь

Три выбранные функции представляют принципиально разные подходы к обучению sentence embeddings:

| Loss | Тип подхода | Ключевая идея |
|------|-------------|---------------|
| **InfoNCE** | Contrastive learning | Сближает positive пары, отталкивает все остальные в batch через softmax |
| **Triplet Loss** | Metric learning | Обучает относительным расстояниям: d(a,n) > d(a,p) + margin |
| **Cosine Similarity Loss** | Regression-based | MSE между cosine_sim(emb1, emb2) и целевым скором схожести |

**Почему именно эти три:**
- InfoNCE концептуально близок к `MultipleNegativesRankingLoss` из sentence-transformers, но рассматривается отдельно из-за явного temperature-параметра и прямого управления in-batch негативами.
- Triplet Loss — классика metric learning, задаёт геометрию пространства через относительные расстояния.
- CosineSimilarityLoss — единственный метод с явным supervision сигналом: либо label-derived (NLI), либо human similarity annotations (STS train).

---

## Supervision strategies

Помимо выбора loss function, сравниваются три стратегии supervision:

| Режим | Датасет | Описание |
|-------|---------|----------|
| `nli_only` | SNLI + MNLI | Entailment/contradiction как суррогатный сигнал схожести |
| `sts_only` | STS train | Human similarity annotations (5749 пар, нормализованы /5.0) |
| `nli_plus_sts` | NLI → STS | Двухэтапное: Stage 1 на NLI, Stage 2 fine-tuning на STS |

**Evaluation protocol (без утечки):**
- `STS train` → для обучения (только в режимах `sts_only` и `nli_plus_sts`)
- `STS dev` → для валидации во время обучения и подбора гиперпараметров
- `STS test` → только для финальной оценки, не используется до конца обучения

---

## Backbone models

| Model | Size | Параметры |
|-------|------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Small | ~22M |
| `bert-base-uncased` | Medium | ~110M |
| `roberta-large` | Large | ~355M |

Stage 1 сравнивает loss functions на `bert-base-uncased`. Stage 2 берёт лучшую конфигурацию и сравнивает backbone sizes.

---

## Установка

```bash
pip install -r requirements.txt
```

---

## Скачивание датасетов

```bash
python datasets/download_datasets.py
```

Для быстрого тестирования (ограничить NLI выборку):

```bash
python datasets/download_datasets.py --max_samples 10000
```

Что скачивается:
- SNLI + MNLI → `data/processed/nli_pairs.pkl`, `nli_triplets.pkl`, `nli_cosine.pkl`
- STS Benchmark (train/val/test) → `data/processed/sts_train.pkl`, `sts_val.pkl`, `sts_test.pkl`

---

## Обучение одного эксперимента

```bash
# InfoNCE на NLI, LR=2e-5
python train.py --config experiments/info_nce_nli.yaml --lr 2e-5

# Cosine на STS train (human annotations)
python train.py --config experiments/cosine_sts.yaml --lr 2e-5

# Two-stage: NLI → STS fine-tuning
python train.py --config experiments/cosine_nli_plus_sts.yaml --lr 2e-5

# С другим backbone
python train.py --config experiments/info_nce_nli.yaml --lr 2e-5 \
    --backbone sentence-transformers/all-MiniLM-L6-v2
```

---

## Запуск всех экспериментов

### Stage 1 — Loss × Supervision strategy × LR sweep

```bash
python run_all_experiments.py
```

С ограничением выборки:

```bash
python run_all_experiments.py --max_train_samples 5000 --skip_existing
```

Выбрать конкретные эксперименты:

```bash
python run_all_experiments.py --experiments info_nce triplet cosine cosine_sts cosine_nli_plus_sts
```

### Stage 2 — Backbone comparison

```bash
python run_all_experiments.py --stage2
```

Запускает лучшую конфигурацию из Stage 1 на всех трёх backbone.

---

## Evaluation

```bash
python evaluate.py \
  --config experiments/info_nce_nli.yaml \
  --checkpoint outputs/info_nce/nli_only/BERT-base/lr_2_0e-05/checkpoints/best_model.pt \
  --plot_embeddings
```

---

## Сравнение результатов

```bash
python compare_results.py
```

Выводит:
1. Лучший Spearman по каждой loss function (NLI-only)
2. Сравнение supervision strategies (NLI / STS / NLI+STS)
3. Сравнение backbone sizes (если запускался Stage 2)
4. Анализ чувствительности к LR
5. Divergence report

Генерирует графики в `outputs/plots/`.

---

## Структура экспериментальных конфигов

```text
experiments/
├── info_nce_nli.yaml       # InfoNCE | NLI only
├── triplet_nli.yaml        # Triplet  | NLI only
├── cosine_nli.yaml         # Cosine   | NLI only
├── cosine_sts.yaml         # Cosine   | STS train (human annotations)
└── cosine_nli_plus_sts.yaml # Cosine  | Two-stage: NLI → STS
```

---

## Структура outputs

```text
outputs/
├── info_nce/nli_only/BERT-base/lr_*/
│   ├── checkpoints/best_model.pt
│   ├── logs/train_log.csv     # epoch, step, train_loss, lr, grad_norm
│   ├── logs/val_log.csv       # epoch, spearman_score, stage, epoch_time_s
│   └── metrics/result.json
├── cosine/sts_only/BERT-base/lr_*/
├── cosine/nli_plus_sts/BERT-base/lr_*/
├── plots/                     # все графики
└── all_results.csv
```

---

## Генерируемые графики

| График | Файл | Описание |
|--------|------|----------|
| Train Loss vs Steps | `train_loss_comparison.png` | Лучший LR на каждый метод |
| Spearman vs Epochs | `spearman_vs_epochs.png` | Сходимость на val set |
| Spearman vs LR | `spearman_vs_lr.png` | LR sensitivity |
| Stability Heatmap | `stability_heatmap.png` | Loss × LR matrix |
| Convergence per Loss | `convergence_{loss}.png` | Все LR на одном графике |
| Time vs Quality | `time_vs_quality.png` | Trade-off scatter |
| Gradient Norms | `grad_norm.png` | Стабильность градиентов |
| Supervision Comparison | `supervision_comparison.png` | NLI / STS / NLI+STS |
| STS Fine-tuning Effect | `sts_finetuning_effect.png` | Delta Spearman от Stage 2 |
| Backbone Comparison | `backbone_comparison.png` | MiniLM / BERT / RoBERTa |
| PCA / t-SNE | `pca_{loss}.png`, `tsne_{loss}.png` | Проекция эмбеддингов |

---

## Исследовательские вопросы

1. Какая loss function даёт лучшую Spearman корреляцию на STS Benchmark?
2. Какая более устойчива к выбору LR (маленький std по LR-grid)?
3. Улучшают ли human similarity annotations (STS train) качество embeddings по сравнению с entailment-only?
4. Что эффективнее: NLI supervision vs human similarity supervision vs двухэтапное?
5. Есть ли дивергенция / коллапс эмбеддингов при больших LR?
6. Как размер backbone влияет на итоговое качество?

---

## Reproducibility

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Фиксируется для каждого запуска через `config.seed`.

---

## Требования к железу

- GPU с CUDA рекомендован (fp16 mixed precision)
- VRAM: минимум 8 GB (уменьшить `batch_size` при нехватке)
- RAM: 16 GB для полного SNLI + MNLI

Без GPU код работает на CPU (медленнее).
