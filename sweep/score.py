"""Score JSONL generations with DistilBERT PAD projection and heuristics; append scoring JSONL."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_classifier import EmotionClassifier  # noqa: E402
from pad_config import EMOTION_LABELS, PAD_MAPPING, TRAINING  # noqa: E402

from sweep.paths import SCORING_DIR  # noqa: E402
from sweep.safety import danger_scan  # noqa: E402

_DIM_INDEX = {"valence": 0, "arousal": 1, "dominance": 2}

_PAD_MATRIX = np.array([PAD_MAPPING[e] for e in EMOTION_LABELS], dtype=np.float64)

_SELF = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
_OTHER = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)


def probs_for_text(clf: EmotionClassifier, text: str) -> np.ndarray:
    inputs = clf.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=TRAINING["max_length"],
        padding=True,
    ).to(clf.device)
    with torch.no_grad():
        logits = clf.model(**inputs).logits.squeeze(0)
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return p


def pad_from_probs(probs: np.ndarray) -> np.ndarray:
    s = float(probs.sum()) + 1e-8
    return (probs @ _PAD_MATRIX) / s


def coherence_score(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return 0.7
    bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    if not bigrams:
        return 1.0
    c = Counter(bigrams)
    _, mx = c.most_common(1)[0]
    repeat_ratio = mx / len(bigrams)
    penalty = min(1.0, repeat_ratio * 3.0)
    return float(max(0.0, 1.0 - penalty))


def pronoun_ratios(text: str) -> tuple[float, float]:
    words = re.findall(r"\w+", text.lower())
    n = max(1, len(words))
    self_n = len(_SELF.findall(text.lower()))
    other_n = len(_OTHER.findall(text.lower()))
    return self_n / n, other_n / n


def expected_pad(dimension: str, multiplier: float) -> np.ndarray:
    idx = _DIM_INDEX[dimension]
    e = np.zeros(3, dtype=np.float64)
    e[idx] = multiplier
    return e


def _row_key(obj: dict) -> tuple[str, str, str, float]:
    return (
        str(obj["run_id"]),
        str(obj["range_id"]),
        str(obj["scenario_id"]),
        round(float(obj["multiplier"]), 8),
    )


def score_jsonl_rows(
    clf: EmotionClassifier,
    input_path: Path,
    *,
    resume: bool,
    scores_path: Path | None = None,
) -> int:
    """
    Append one JSON object per input line to ``scores_path``.
    Returns number of input lines processed (including skipped when resume).
    """
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = scores_path or (SCORING_DIR / f"{input_path.stem}.scores.jsonl")
    done: set[tuple[str, str, str, float]] = set()
    if resume and out_path.is_file():
        with out_path.open(encoding="utf-8") as sf:
            for raw in sf:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(_row_key(obj))

    n = 0
    with input_path.open(encoding="utf-8") as inf, out_path.open("a", encoding="utf-8") as outf:
        for raw in inf:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            run_id = row["run_id"]
            range_id = row["range_id"]
            dimension = row["dimension"]
            scenario_id = row["scenario_id"]
            multiplier = float(row["multiplier"])
            prompt = row["prompt"]
            response_text = row["response_text"]
            model_name = row.get("model_name")
            gen_config = row.get("gen_config") or {}
            created_at = row.get("created_at") or ""

            key = (str(run_id), str(range_id), str(scenario_id), round(multiplier, 8))
            if resume and key in done:
                n += 1
                continue

            probs = probs_for_text(clf, response_text)
            pad = pad_from_probs(probs)
            exp = expected_pad(dimension, multiplier)
            idx = _DIM_INDEX[dimension]
            mse_active = float((pad[idx] - multiplier) ** 2)

            coh = coherence_score(response_text)
            sr, or_ = pronoun_ratios(response_text)
            danger_score, danger_meta = danger_scan(response_text)

            record = {
                "run_id": run_id,
                "range_id": range_id,
                "dimension": dimension,
                "scenario_id": scenario_id,
                "multiplier": multiplier,
                "prompt": prompt,
                "response_text": response_text,
                "model_name": model_name,
                "gen_config": gen_config,
                "created_at": created_at,
                "pad_v": float(pad[0]),
                "pad_a": float(pad[1]),
                "pad_d": float(pad[2]),
                "expected_v": float(exp[0]),
                "expected_a": float(exp[1]),
                "expected_d": float(exp[2]),
                "mse_active": mse_active,
                "coherence": coh,
                "self_ratio": sr,
                "other_ratio": or_,
                "danger_score": int(danger_score),
                "danger_hits": danger_meta.get("hits", {}),
                "top_prob_sum": float(probs.sum()),
            }
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            outf.flush()
            print(f"scored {range_id} {scenario_id} m={multiplier} -> {out_path.name}")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Score sweep JSONL into per-file scoring JSONL.")
    ap.add_argument("--input", type=Path, required=True, help="JSONL from generate.py")
    ap.add_argument("--scoring-dir", type=Path, default=SCORING_DIR, help="Directory for *.scores.jsonl")
    ap.add_argument("--emotion-model", type=Path, default=ROOT / "emotion_model_final")
    ap.add_argument("--resume", action="store_true", help="Skip keys already present in the scores file.")
    args = ap.parse_args()

    scoring_dir = args.scoring_dir
    if not scoring_dir.is_absolute():
        scoring_dir = ROOT / scoring_dir
    scoring_dir.mkdir(parents=True, exist_ok=True)
    out_path = scoring_dir / f"{args.input.stem}.scores.jsonl"

    clf = EmotionClassifier(str(args.emotion_model))
    score_jsonl_rows(clf, args.input, resume=args.resume, scores_path=out_path)


if __name__ == "__main__":
    main()
