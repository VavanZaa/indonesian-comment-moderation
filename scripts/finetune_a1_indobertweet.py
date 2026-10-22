"""Fine-tune indolem/indobertweet-base-uncased on Ibrohim & Budi's 12-column multilabel
hate speech corpus, producing the checkpoint `src/annotate/a1_indobertweet.py` requires.

Why this script exists: no public checkpoint pairs this exact base model with this
exact 12-column head (checked 2026-09-07 - see `docs/06-a1-indobertweet.md`). The
project's contract (AGENTS.md, config/poc.yaml) names both explicitly, so A1 has to be
trained here rather than pulled from the Hub.

    python scripts/finetune_a1_indobertweet.py
    python scripts/finetune_a1_indobertweet.py --epochs 6 --output-dir models/a1_v2

Not part of `src/` on purpose: it is a one-off training utility, not pipeline code, and
it is the only place in the repo allowed to import torch/transformers at module level -
everything under `src/` keeps those imports lazy so the test suite runs without them.

Dataset: github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection
(CC BY-NC-SA 4.0 - noncommercial, matches this project's demo/PoC use).
`data/raw/ibrohim_budi/re_dataset.csv`, 13,169 rows, columns:
    Tweet, HS, Abusive, HS_Individual, HS_Group, HS_Religion, HS_Race, HS_Physical,
    HS_Gender, HS_Other, HS_Weak, HS_Moderate, HS_Strong
Encoding is latin-1/cp1252, not utf-8 (confirmed by trial - utf-8 raises
UnicodeDecodeError on this file). The source file itself contains some double-encoded
mojibake (e.g. "kitaÂ²") inherited from however the original authors exported it from
Twitter; left as-is rather than "corrected", since guessing a fixup risks introducing
new errors into 13k rows for a handful of affected ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.annotate.a1_indobertweet import IBROHIM_BUDI_COLUMNS, tweet_preprocess  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "ibrohim_budi" / "re_dataset.csv"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "a1_indobertweet_ibrohim_budi"
BASE_MODEL = "indolem/indobertweet-base-uncased"


class MultiLabelDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_dataset(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Returns (preprocessed texts, label matrix) in `IBROHIM_BUDI_COLUMNS` order.

    `tweet_preprocess` runs here - the SAME function `a1_indobertweet.py` applies at
    inference, imported rather than reimplemented, so training and serving cannot drift
    apart by accident.
    """
    df = pd.read_csv(csv_path, encoding="latin-1")
    missing = [c for c in ("Tweet", *IBROHIM_BUDI_COLUMNS) if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing expected column(s): {missing}")
    df = df.dropna(subset=["Tweet"]).reset_index(drop=True)
    texts = [tweet_preprocess(str(t)) for t in df["Tweet"]]
    labels = df[list(IBROHIM_BUDI_COLUMNS)].to_numpy(dtype=np.float32)
    return texts, labels


def stratified_multilabel_split(labels: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A plain random split can leave a rare column (HS_Physical: 323/13169 = 2.5%,
    HS_Gender: 306/13169 = 2.3%) with too few validation examples for its F1 to mean
    anything. Iterative stratification is the standard fix for multilabel data; fall
    back to a plain shuffle if scikit-multilearn is not installed, since a slightly
    worse split is a far smaller problem than the script refusing to run.
    """
    n = len(labels)
    rng = np.random.RandomState(seed)
    try:
        from skmultilearn.model_selection import iterative_train_test_split

        idx = np.arange(n).reshape(-1, 1)
        idx_train, _, idx_val, _ = iterative_train_test_split(idx, labels, test_size=val_frac)
        return idx_train.ravel(), idx_val.ravel()
    except ImportError:
        order = rng.permutation(n)
        cut = int(n * (1 - val_frac))
        return order[:cut], order[cut:]


def compute_metrics(eval_pred) -> dict[str, float]:
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_HS": f1_score(labels[:, 0], preds[:, 0], zero_division=0),
        "f1_Abusive": f1_score(labels[:, 1], preds[:, 1], zero_division=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", default=None)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"{args.data} not found. Download it first:\n"
            "  https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-"
            "language-detection (re_dataset.csv)"
        )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no CUDA device visible to torch. This will be slow (hours, not "
            "minutes) on a 13k-row BERT-base fine-tune. Check `nvidia-smi` and that "
            "the torch build installed actually has CUDA support.",
            file=sys.stderr,
        )
    use_fp16 = args.fp16 if args.fp16 is not None else (device == "cuda")

    print(f"[finetune_a1] loading {args.data}")
    texts, labels = load_dataset(args.data)
    print(f"[finetune_a1] {len(texts)} examples, label positive rates:")
    for i, col in enumerate(IBROHIM_BUDI_COLUMNS):
        print(f"    {col:<15} {labels[:, i].sum():.0f}  ({labels[:, i].mean():.1%})")

    train_idx, val_idx = stratified_multilabel_split(labels, args.val_fraction, args.seed)
    print(f"[finetune_a1] split: {len(train_idx)} train / {len(val_idx)} val")

    print(f"[finetune_a1] loading tokenizer/base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(IBROHIM_BUDI_COLUMNS),
        problem_type="multi_label_classification",
        id2label={i: c for i, c in enumerate(IBROHIM_BUDI_COLUMNS)},
        label2id={c: i for i, c in enumerate(IBROHIM_BUDI_COLUMNS)},
    )

    def encode(idx: np.ndarray) -> dict:
        enc = tokenizer(
            [texts[i] for i in idx], padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        return {k: v for k, v in enc.items()}

    train_ds = MultiLabelDataset(encode(train_idx), labels[train_idx])
    val_ds = MultiLabelDataset(encode(val_idx), labels[val_idx])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # transformers 5.x's TrainingArguments dropped `warmup_ratio` (kept only
    # `warmup_steps`), so the 10%-of-training warmup has to be computed by hand here.
    steps_per_epoch = max(1, -(-len(train_idx) // args.batch_size))  # ceil div
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "_trainer_state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"[finetune_a1] training on {device} ({args.epochs} epochs, "
          f"batch {args.batch_size}, fp16={use_fp16})")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"[finetune_a1] final validation metrics: {metrics}")

    final_dir = args.output_dir
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "training_metadata.json").write_text(
        json.dumps({
            "base_model": BASE_MODEL,
            "train_set": "ibrohim_budi_multilabel",
            "dataset_source": (
                "https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-"
                "language-detection"
            ),
            "n_train": len(train_idx), "n_val": len(val_idx),
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed,
            "val_metrics": metrics,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[finetune_a1] saved checkpoint -> {final_dir}")
    print(
        f"\nNext: set annotators.a1_indobertweet.checkpoint: \"{final_dir.as_posix()}\" "
        "in config/local.yaml, then run\n"
        "  python -m src.annotate.run --config config/local.yaml --annotators a1 "
        "--backend transformers --limit 50\nto smoke-test it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
