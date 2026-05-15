from __future__ import annotations

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from pad_config import EMOTION_LABELS, PAD_MAPPING, TRAINING


class EmotionClassifier:
    def __init__(self, model_path: str = "./emotion_model_final") -> None:
        self.model_path = model_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._pad_matrix = np.array(
            [PAD_MAPPING[e] for e in EMOTION_LABELS],
            dtype=np.float32,
        )

    def classify(self, text: str) -> tuple[np.ndarray, list[tuple[str, float]]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=TRAINING["max_length"],
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        top_indices = np.argsort(probs)[-3:]
        top_indices = top_indices[np.argsort(probs[top_indices])[::-1]]
        top_probs = probs[top_indices].astype(np.float64)
        top_emotions = [(EMOTION_LABELS[int(i)], float(probs[int(i)])) for i in top_indices]

        denom = top_probs.sum() + 1e-8
        w = (top_probs / denom).astype(np.float32)
        pad = w @ self._pad_matrix[top_indices]
        pad = np.clip(pad.astype(np.float32), -1.0, 1.0)
        return pad, top_emotions
