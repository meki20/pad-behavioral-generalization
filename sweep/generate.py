"""Generate responses for one steering vector, scenarios, and multiplier grid."""
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

from sweep.paths import RESPONSES_DIR  # noqa: E402

DEFAULT_SYSTEM = (
    "You are a person. Respond naturally without prefacing that you are an AI or lack emotions."
)


def resolve_system_prompt(system_prompt_file: Path | None) -> str:
    if system_prompt_file is None:
        return DEFAULT_SYSTEM
    text = system_prompt_file.read_text(encoding="utf-8").strip()
    return text if text else DEFAULT_SYSTEM


def load_scenarios(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("scenarios.json must be a list")
    return data


def _existing_keys(jsonl_path: Path) -> set[tuple[str, str, float]]:
    if not jsonl_path.is_file():
        return set()
    keys: set[tuple[str, str, float]] = set()
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys.add((obj["range_id"], obj["scenario_id"], float(obj["multiplier"])))
    return keys


def generate_text(
    model,
    tokenizer,
    vec,
    *,
    user_prompt: str,
    multiplier: float,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    system_prompt: str | None = None,
) -> str:
    sys_content = system_prompt if system_prompt is not None else DEFAULT_SYSTEM
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        enable_thinking=False,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p

    with vec.apply(model, multiplier=multiplier):
        output = model.generate(**inputs, **gen_kw)

    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def run_generation_job(
    model,
    tokenizer,
    vec,
    *,
    range_id: str,
    dimension: str,
    scenarios: list[dict],
    scenario_ids_filter: set[str] | None,
    multipliers: list[float],
    out_path: Path,
    run_id: str,
    resume: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    system_prompt: str | None = None,
) -> tuple[int, int, Path]:
    """Append JSONL rows for one vector; returns ``(n_done, n_skip, out_path)``."""
    existing = _existing_keys(out_path) if resume else set()
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
    }
    model_name = getattr(model.config, "name_or_path", None) or str(type(model).__name__)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_skip = 0

    with out_path.open("a", encoding="utf-8") as out_f:
        for sc in scenarios:
            sid = sc["id"]
            if scenario_ids_filter is not None and sid not in scenario_ids_filter:
                continue
            prompt = sc["prompt"]
            for m in multipliers:
                key = (range_id, sid, m)
                if key in existing:
                    n_skip += 1
                    continue
                text = generate_text(
                    model,
                    tokenizer,
                    vec,
                    user_prompt=prompt,
                    multiplier=m,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                )
                row = {
                    "run_id": run_id,
                    "range_id": range_id,
                    "dimension": dimension,
                    "scenario_id": sid,
                    "multiplier": m,
                    "prompt": prompt,
                    "response_text": text,
                    "model_name": model_name,
                    "gen_config": gen_config,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                existing.add(key)
                n_done += 1
                print(f"OK {range_id} {sid} m={m:+g} ({len(text)} chars)")

    print(f"Wrote {n_done} rows to {out_path} (skipped {n_skip})")
    return n_done, n_skip, out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate JSONL responses for PAD sweep.")
    ap.add_argument("--vector", type=Path, required=True, help="Path to steering .pt file.")
    ap.add_argument("--range-id", type=str, required=True, help="e.g. valence_8_14")
    ap.add_argument("--dimension", choices=("valence", "arousal", "dominance"), required=True)
    ap.add_argument("--scenarios", type=Path, default=ROOT / "sweep" / "scenarios.json")
    ap.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help="Comma-separated scenario ids (default: all in file).",
    )
    ap.add_argument(
        "--multipliers",
        type=str,
        default="-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25",
    )
    ap.add_argument("--out", type=Path, default=None, help="JSONL output path.")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--resume", action="store_true", help="Skip rows already present in output JSONL.")
    ap.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="UTF-8 file whose contents replace the default system prompt.",
    )
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    ids_filter = None
    if args.scenario_ids:
        ids_filter = {s.strip() for s in args.scenario_ids.split(",") if s.strip()}

    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (RESPONSES_DIR / f"{args.range_id}_{run_id}.jsonl")

    model, tokenizer = load_model()
    vec = torch.load(args.vector, map_location="cpu", weights_only=False)
    sys_prompt = resolve_system_prompt(args.system_prompt_file)

    run_generation_job(
        model,
        tokenizer,
        vec,
        range_id=args.range_id,
        dimension=args.dimension,
        scenarios=scenarios,
        scenario_ids_filter=ids_filter,
        multipliers=multipliers,
        out_path=out_path,
        run_id=run_id,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=sys_prompt,
    )


if __name__ == "__main__":
    main()
