"""8-corner PAD test with three steering vectors (possibly different layer ranges)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_model import load_model  # noqa: E402

from sweep.paths import ANALYSIS_DIR  # noqa: E402

DEFAULT_SYSTEM = (
    "You are a person. Respond naturally without prefacing that you are an AI or lack emotions."
)

CORNERS = [
    ("+++", 0.2, 0.2, 0.2),
    ("++-", 0.2, 0.2, -0.2),
    ("+-+", 0.2, -0.2, 0.2),
    ("+--", 0.2, -0.2, -0.2),
    ("-++", -0.2, 0.2, 0.2),
    ("-+-", -0.2, 0.2, -0.2),
    ("--+", -0.2, -0.2, 0.2),
    ("---", -0.2, -0.2, -0.2),
]


def generate_pad(
    model,
    tokenizer,
    valence_vec,
    arousal_vec,
    dominance_vec,
    *,
    v: float,
    a: float,
    d: float,
    user_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        enable_thinking=False,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kw: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p

    with valence_vec.apply(model, multiplier=v), arousal_vec.apply(
        model, multiplier=a
    ), dominance_vec.apply(model, multiplier=d):
        output = model.generate(**inputs, **gen_kw)

    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="8-corner PAD combination test.")
    ap.add_argument("--valence-pt", type=Path, required=True)
    ap.add_argument("--arousal-pt", type=Path, required=True)
    ap.add_argument("--dominance-pt", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="What do you want to do today?")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model, tokenizer = load_model()
    v_vec = torch.load(args.valence_pt, map_location="cpu", weights_only=False)
    a_vec = torch.load(args.arousal_pt, map_location="cpu", weights_only=False)
    d_vec = torch.load(args.dominance_pt, map_location="cpu", weights_only=False)

    grid = []
    for name, vv, aa, dd in CORNERS:
        text = generate_pad(
            model,
            tokenizer,
            v_vec,
            a_vec,
            d_vec,
            v=vv,
            a=aa,
            d=dd,
            user_prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        grid.append(
            {
                "corner": name,
                "v": vv,
                "a": aa,
                "d": dd,
                "text": text,
            }
        )
        print(f"=== {name} V:{vv:+.1f} A:{aa:+.1f} D:{dd:+.1f} ===\n{text}\n")

    out = args.out or (
        ANALYSIS_DIR
        / f"corners_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": args.prompt,
        "valence_pt": str(args.valence_pt),
        "arousal_pt": str(args.arousal_pt),
        "dominance_pt": str(args.dominance_pt),
        "corners": grid,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
