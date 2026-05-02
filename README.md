# Embedding Fine-Tuning Playground

This repository is a small experiment for fine-tuning a text embedding model with `sentence-transformers` on the Natural Questions pair dataset.

The overall fine-tuning workflow in this project was learned from: https://sbert.net/docs/sentence_transformer/training_overview.html.

The current workflow:

1. Download the `sentence-transformers/natural-questions` dataset.
2. Split it into train, eval, and test sets.
3. Fine-tune an embedding model with `MultipleNegativesRankingLoss`.
4. Save checkpoints and the final model locally.

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── split_data.py
└── training.py
```

## What Each File Does

- `split_data.py`: Downloads the Natural Questions dataset and creates reusable train, eval, and test splits. The resulting files are saved in `dataset/`.
- `training.py`: Loads the saved dataset splits, fine-tunes an embedding model, evaluates it, and saves checkpoints and the final model in `nq-e5-large-instruct/`.

## Dataset

The training data comes from:

- Hugging Face dataset: `sentence-transformers/natural-questions`

Based on the saved dataset metadata in this repo, examples contain:

- `query`
- `answer`

`split_data.py` performs:

- 70% train
- 27% eval
- 3% test

This happens by first splitting `train` into `70/30`, then splitting the `30%` portion into `90/10`.

## Training Configuration

`training.py` currently uses:

- Base model: `intfloat/multilingual-e5-large-instruct`
- Loss: `MultipleNegativesRankingLoss`
- Epochs: `10`
- Train batch size: `16`
- Eval batch size: `16`
- Learning rate: `1e-10`
- Precision: `bf16=True`
- Early stopping patience: `3`

### Notes
- Since the dataset is in the form of `(anchor, positive)` pairs, `MultipleNegativesRankingLoss` is a good fit for this training setup based on the source: https://sbert.net/docs/sentence_transformer/loss_overview.html.
- `batch_sampler=BatchSamplers.NO_DUPLICATES` helps ensure that no duplicate entries appear in a single batch. This is important for `MultipleNegativesRankingLoss` because it uses other items in the batch as negative examples.
- `bf16` is only supported on newer hardware, such as NVIDIA Ampere GPUs like the A100, 3090, and 4090, or on Google TPUs. If you are using an older GPU such as a T4, use `fp16` instead.

## Hardware Requirement

Run training on a GPU with at least `24 GB` of VRAM.

## How To Run

1. Install the dependencies first.

```bash
uv sync
```

2. Run `split_data.py` to download and split the dataset.

```bash
python split_data.py
```

3. Run `training.py` to start fine-tuning.

```bash
python training.py
```
