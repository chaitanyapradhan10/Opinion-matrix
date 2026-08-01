"""
train_transformer.py
----------------------
Bonus phase of the Sentiment Analysis pipeline: fine-tune a pretrained
Transformer (DistilBERT / BERT / RoBERTa) as a stronger alternative to the
classical TF-IDF + ML models trained in train_model.py.

Unlike the classical pipeline, Transformers work best on lightly-cleaned,
near-raw text (they have their own subword tokenizer and rely on casing/
punctuation/word order for context) - so this script uses a separate, minimal
cleaning step instead of the stemmed/lemmatized `clean_review` column.

Run:
    python train_transformer.py
    python train_transformer.py --model bert-base-uncased --epochs 4
    python train_transformer.py --model roberta-base

Outputs (written to ../models/transformer/):
    pytorch_model.bin / model.safetensors + config.json   (fine-tuned model)
    tokenizer files (vocab, tokenizer_config.json, etc.)
    label_encoder.pkl                                     (class <-> id map)
    evaluation_report.txt                                 (metrics)
"""

import argparse
import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

#sys.path.append(".")
from data_preprocessing import load_data
from logging_config import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42

# Any of these work out of the box - swap via --model:
#   distilbert-base-uncased   (fast, ~66M params, good default)
#   bert-base-uncased         (~110M params, classic baseline)
#   roberta-base               (~125M params, usually the strongest of the three)
DEFAULT_MODEL = "distilbert-base-uncased"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")


# ---------------------------------------------------------------------------
# Minimal cleaning - keep casing/punctuation, only strip noise the tokenizer
# can't usefully interpret (HTML tags, raw URLs, extra whitespace).
# ---------------------------------------------------------------------------
def light_clean(text: str) -> str:
    text = str(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ReviewDataset(Dataset):
    """Wraps tokenized encodings + labels for the HuggingFace Trainer."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a pretrained Transformer for sentiment classification.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                    help="HuggingFace checkpoint: distilbert-base-uncased | bert-base-uncased | roberta-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--data", default="../data/reviews.csv")
    p.add_argument("--out-dir", default="../models/transformer")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    report_lines = [f"Model: {args.model}", f"Epochs: {args.epochs}", f"Max length: {args.max_length}"]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {args.model} | Epochs: {args.epochs} | Max length: {args.max_length}")

    # Phase 3 + light cleaning
    logger.info(f"Loading data from {args.data}")
    import os

    print("Dataset path:", args.data)
    print("Exists:", os.path.exists(args.data))
    df = load_data(args.data)
    df["clean_review"] = df["review"].apply(light_clean)
    logger.info(f"Loaded {len(df)} rows")

    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])
    num_labels = len(le.classes_)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"].tolist(), y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # carve a small validation split out of train for early stopping
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )

    logger.info(f"Train: {len(X_train_text)} | Val: {len(X_val_text)} | Test: {len(X_test_text)}")

    logger.info(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={i: c for i, c in enumerate(le.classes_)},
        label2id={c: i for i, c in enumerate(le.classes_)},
    )
    logger.info(f"Model loaded with {num_labels} labels: {list(le.classes_)}")

    def tokenize(texts):
        return tokenizer(
            texts, truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt"
        )

    train_dataset = ReviewDataset(tokenize(X_train_text), y_train)
    val_dataset = ReviewDataset(tokenize(X_val_text), y_val)
    test_dataset = ReviewDataset(tokenize(X_test_text), y_test)

    training_args = TrainingArguments(
        output_dir="../models/transformer_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning complete.")

    # Phase 8: evaluate on the held-out test set
    logger.info("Evaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)

    logger.info(f"=== {args.model} (fine-tuned) ===")
    logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    report_lines += [
        f"\n=== {args.model} (fine-tuned) ===",
        f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}",
        f"Confusion Matrix:\n{cm}",
        report,
    ]

    # Phase 10: save model + tokenizer + label encoder
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    joblib.dump(le, os.path.join(args.out_dir, "label_encoder.pkl"))

    with open(os.path.join(args.out_dir, "evaluation_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved fine-tuned model, tokenizer, and label encoder to {args.out_dir}/")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training run failed with an unhandled exception.")
        raise