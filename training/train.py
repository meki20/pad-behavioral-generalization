import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TRAINING_DIR = Path(__file__).resolve().parent

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from pad_config import EMOTION_LABELS, NUM_LABELS, TRAINING as _TRAINING_BASE

TRAINING = {
    **_TRAINING_BASE,
    "output_dir": str(_TRAINING_DIR / "checkpoints"),
}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]).float(),
    }


def load_and_prepare():
    from datasets import Dataset, DatasetDict

    dataset = load_dataset("go_emotions", "raw")
    emotion_cols = EMOTION_LABELS

    def aggregate_split(split):
        df = split.to_pandas()
        agg = df.groupby("id").agg(
            {**{e: "max" for e in emotion_cols}, "text": "first"}
        ).reset_index(drop=True)
        return Dataset.from_pandas(agg, preserve_index=False)

    dataset = DatasetDict(
        {split: aggregate_split(dataset[split]) for split in dataset.keys()}
    )

    def to_multihot(batch):
        labels = np.zeros((len(batch["text"]), NUM_LABELS), dtype=np.float32)
        for i, emotion in enumerate(emotion_cols):
            for j, val in enumerate(batch[emotion]):
                labels[j][i] = float(val)
        return {"labels": labels}

    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})
    dataset = dataset.map(
        to_multihot,
        batched=True,
        remove_columns=[
            col
            for col in dataset["train"].column_names
            if col not in ("text", "labels")
        ],
    )
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=TRAINING["max_length"],
        )

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= TRAINING["threshold"]).astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, average="micro")
    except ValueError:
        auc = 0.0

    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "auc": auc}


def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained(TRAINING["model_name"])

    print("Loading dataset...")
    dataset = load_and_prepare()

    print("Tokenizing...")
    dataset = tokenize_dataset(dataset, tokenizer)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = DistilBertForSequenceClassification.from_pretrained(
        TRAINING["model_name"],
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    log_dir = _TRAINING_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(TRAINING["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=TRAINING["output_dir"],
        num_train_epochs=TRAINING["epochs"],
        per_device_train_batch_size=TRAINING["batch_size"],
        per_device_eval_batch_size=TRAINING["batch_size"],
        learning_rate=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        logging_dir=str(log_dir),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    print("Training...")
    trainer.train()

    final_dir = _REPO_ROOT / "emotion_model_final"
    print(f"Saving model to {final_dir}...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("Done.")


if __name__ == "__main__":
    main()
