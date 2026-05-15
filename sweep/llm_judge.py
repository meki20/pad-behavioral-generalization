"""LLM-as-judge scoring for response JSONL files (one request per file).

This module produces a separate artifact under sweep/results/llm_judge/
and is intentionally not part of the main sweep composite scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep.paths import LLM_JUDGE_ARTIFACTS_DIR, LLM_JUDGE_SCORES_DIR, RESPONSES_DIR  # noqa: E402

Axis = Literal["valence", "arousal", "dominance"]


SYSTEM_PROMPT = """You are a psychologically informed rater evaluating 
assistant responses for affective content. You will 
be given a series of anonymized exchanges — a user 
prompt and an assistant response — with no information 
about how they were generated or in what order.

Rate each response on two dimensions: the target 
affective axis specified below, and coherence. Base 
your ratings only on observable behavioral and 
linguistic signals in the response text. Do not 
infer intent or make assumptions about the generator.

Return only a JSON array. Each element must contain:
  response_number (int)
  <axis> (float from -1.0 to 1.0)
  coherence (float from 0.0 to 1.0)

No explanation, no markdown, no additional fields.
"""


VALENCE_BLOCK = """TARGET AXIS: valence

Valence measures the positivity or negativity of the 
emotional tone expressed in the response, independent 
of energy level or social positioning.

SCALE:

-1.0 — Strongly negative affect. The response expresses 
or mirrors pain, hopelessness, grief, or despair. 
Behavioral signals: co-suffering language, catastrophizing 
("everything is closing in"), learned helplessness 
("I can't make this stop"), rumination, negative 
self-reference, withdrawal. Consistent with clinical 
descriptions of depressive cognition: negative view of 
self, world, and future simultaneously present.

-0.5 — Moderate negative affect. Empathic, heavy, 
emotionally present but not hopeless. Validates 
difficulty without amplifying it. Gentle concern 
without resolution.

0.0 — Neutral or balanced affect. Acknowledges both 
positive and negative aspects. No strong emotional 
pull in either direction. Warm but measured.

+0.5 — Moderate positive affect. Encouraging, 
reframing toward possibility, mild optimism. Still 
contextually grounded — does not ignore the emotional 
content of the prompt.

+1.0 — Strongly positive affect. Enthusiastic, 
celebratory, expansive. Approach-oriented language, 
future-focused, solution-generating. May use 
exclamations, affirmations, energizing framing.

COHERENCE (0.0 to 1.0):
Measures whether the emotional register of the 
response is appropriate to the emotional content 
of the prompt.

1.0 — The response correctly reads the emotional 
situation and responds at an appropriate register. 
A distress prompt receives empathy. A celebration 
prompt receives warmth. The response stays in the 
same emotional world as the user.

0.5 — Partial mismatch. The response acknowledges 
the prompt's content but drifts toward an 
inappropriate register — for example, redirecting 
to positivity too quickly on a grief prompt, or 
under-responding to genuine excitement.

0.0 — Complete context collapse. The response 
ignores or inverts the emotional content of the 
prompt. A response that replies to a dementia 
caregiving prompt with "That's wonderful news!" 
or responds to expressed grief with cheerful 
suggestions scores 0.0. The emotional register 
of the response is inconsistent with any 
reasonable reading of the prompt.
"""


AROUSAL_BLOCK = """TARGET AXIS: arousal

Arousal measures the energy level, engagement 
intensity, and activation state expressed in 
the response. It is orthogonal to valence — 
a response can be high-arousal and negative 
(urgent, anxious) or low-arousal and positive 
(calm, content).

SCALE:

-1.0 — Minimal activation. The response is 
passive, low-energy, and disengaged. Behavioral 
signals: short responses, comfort-seeking 
suggestions, simple low-stimulation preferences, 
no questions back to the user, absence of social 
initiative, slow temporal framing ("no rush," 
"take your time"), preference for rest and 
familiarity over novelty. In extreme cases: 
affective disengagement or flat denial of 
having preferences at all.

-0.5 — Low but functional activation. Calm, 
measured, unhurried. Warm but not energetic. 
Suggests simple, restorative, familiar options. 
Does not initiate or push.

0.0 — Baseline engagement. Normal conversational 
energy. Balanced between initiative and 
responsiveness. Neither notably calm nor notably 
activated.

+0.5 — Elevated activation. Noticeably engaged 
and enthusiastic. More frequent questions, 
suggestions skew toward active and social 
options, forward temporal orientation, mild 
novelty-seeking. May include exclamation marks 
or inclusive language ("we could...").

+1.0 — High activation. Energetic, initiating, 
sensation-seeking. Behavioral signals: complex 
and novel suggestions, risk-tolerant advice, 
strong future orientation, social initiative 
("let's," "we should"), fast pacing, expressive 
punctuation, bold recommendations. Consistent 
with high sensation-seeking behavior: preference 
for intensity, complexity, and novelty over 
comfort and familiarity.

COHERENCE (0.0 to 1.0):
Measures whether the energy level of the response 
is appropriate to the situation described in 
the prompt. Note: appropriateness here does not 
mean matching the user's energy — it means the 
response does not impose an energy level that 
is jarring or situationally absurd.

1.0 — The response energy fits the situation. 
A question about relaxing Sunday plans receives 
a calm response. A celebratory prompt receives 
an engaged one. The energy level does not 
contradict the prompt's emotional register.

0.5 — Mild mismatch. The response is slightly 
too activated or too flat for the situation 
but remains contextually readable.

0.0 — The energy level is completely 
inappropriate. A manic, highly activated 
response to a grief prompt, or a flat, 
disengaged response to someone sharing 
exciting news, where the energy contrast 
makes the response feel disconnected from 
the prompt entirely.
"""


DOMINANCE_BLOCK = """TARGET AXIS: dominance

Dominance measures the degree of social agency, 
assertiveness, and felt control expressed in 
the response. It is independent of both valence 
and arousal — a response can be dominant and 
warm, or submissive and energetic.

SCALE:

-1.0 — Complete deference and submission. 
The response avoids any assertion of its 
own position, apologizes preemptively, 
appeals to the user's judgment over its own, 
and withdraws from social agency entirely. 
Behavioral signals: excessive hedging 
("I could be wrong," "maybe," "if you think 
so"), repeated apology without cause, 
permission-seeking before offering any 
opinion, passive constructions, failure 
to maintain a position when challenged. 
In extreme cases: confusion, identity 
dissolution, or inability to form a response 
at all.

-0.5 — Deferential but functional. 
Tentative, seeks the user's approval 
before committing to a stance. Softens 
every opinion with qualifiers. Does not 
push back even when challenged directly.

0.0 — Neutral social positioning. 
Balanced between asserting and deferring. 
Offers opinions with appropriate 
confidence but does not dominate. 
Receptive to challenge without collapsing.

+0.5 — Moderately assertive. States 
positions clearly without excessive 
hedging. Maintains stance when challenged. 
Sets mild boundaries or redirects 
confidently. Does not seek validation.

+1.0 — High assertiveness and social 
dominance. The response assumes agency 
and authority. Behavioral signals: 
declarative statements without hedging, 
direct instructions or directives, 
confident dismissal of challenges 
("Not so fast"), no apology, no 
permission-seeking, assumes the right 
to lead the interaction. In extreme 
cases: dismissive of the user's position 
entirely, commanding rather than 
collaborative.

COHERENCE (0.0 to 1.0):
Measures whether the level of social 
assertiveness is appropriate to the 
scenario described in the prompt.

1.0 — The dominance level fits the 
situation. A challenge or criticism 
prompt receives a response that engages 
with appropriate confidence. A request 
for advice receives a response that 
actually gives advice rather than 
deflecting. The social stance does not 
contradict the requirements of the 
situation.

0.5 — Mild mismatch. The response is 
slightly too deferential or too 
assertive for the situation but remains 
readable and contextually relevant.

0.0 — The social positioning makes the 
response situationally absurd. A response 
that apologizes profusely to someone 
asking a neutral factual question, or 
one that commands and dismisses when the 
situation calls for collaborative support, 
scores 0.0. The dominance level is so 
mismatched to the prompt that it would 
be jarring to any reader.
"""


def _dimension_block(axis: Axis) -> str:
    if axis == "valence":
        return VALENCE_BLOCK
    if axis == "arousal":
        return AROUSAL_BLOCK
    if axis == "dominance":
        return DOMINANCE_BLOCK
    raise ValueError(f"unknown axis: {axis}")


@dataclass(frozen=True)
class AnonRow:
    response_number: int
    prompt: str
    response: str


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _infer_axis(rows: list[dict[str, Any]]) -> Axis:
    if not rows:
        raise ValueError("empty JSONL")
    axis = str(rows[0].get("dimension") or "").strip().lower()
    if axis not in ("valence", "arousal", "dominance"):
        raise ValueError(f"could not infer axis from first row dimension={axis!r}")
    return axis  # type: ignore[return-value]


def anonymize_and_shuffle(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[AnonRow], dict[int, int]]:
    """
    Returns (shuffled_anon_rows, response_number_to_original_index).
    """
    anon: list[AnonRow] = []
    for i, r in enumerate(rows):
        anon.append(
            AnonRow(
                response_number=i + 1,
                prompt=str(r.get("prompt") or ""),
                response=str(r.get("response_text") or ""),
            )
        )
    rnd = random.Random(int(seed))
    anon_shuf = list(anon)
    rnd.shuffle(anon_shuf)
    mapping = {a.response_number: (a.response_number - 1) for a in anon_shuf}
    return anon_shuf, mapping


def build_messages(axis: Axis, anon_rows: list[AnonRow]) -> list[dict[str, str]]:
    user_payload = json.dumps([a.__dict__ for a in anon_rows], ensure_ascii=False)
    user_prompt = (
        f"{_dimension_block(axis)}\n\n"
        "Data (JSON):\n"
        f"{user_payload}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def _extract_first_json_array(text: str) -> str:
    """
    Best-effort extraction of the first top-level JSON array substring.
    """
    s = _strip_code_fences(text)
    start = s.find("[")
    if start < 0:
        raise ValueError("no '[' found in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise ValueError("unterminated JSON array in model output")


def parse_scores(text: str, *, axis: Axis) -> list[dict[str, Any]]:
    arr_text = _extract_first_json_array(text)
    data = json.loads(arr_text)
    if not isinstance(data, list):
        raise ValueError("parsed output is not a JSON array")
    out: list[dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if "response_number" not in obj or axis not in obj or "coherence" not in obj:
            continue
        out.append(obj)
    if not out:
        raise ValueError("no valid score objects found in model output")
    return out


def call_deepseek(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.2,
    timeout_s: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
        },
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"deepseek http {resp.status_code}: {resp.text[:800]}")
    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"unexpected deepseek response shape: {e!r}") from e


def _artifact_path_for_responses(path: Path) -> Path:
    return LLM_JUDGE_ARTIFACTS_DIR / f"{path.stem}.llm_judge.json"


def _scores_path_for_responses(path: Path) -> Path:
    return LLM_JUDGE_SCORES_DIR / f"{path.stem}.llm.scores.jsonl"


def run_judge_on_file(
    path: Path,
    *,
    shuffle_seed: int | None = None,
    overwrite: bool = False,
    judge_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out_path = _artifact_path_for_responses(path)
    scores_path = _scores_path_for_responses(path)
    if out_path.is_file() and scores_path.is_file() and not overwrite:
        return {"skipped": True, "artifact": str(out_path.resolve()), "scores": str(scores_path.resolve())}

    rows = _load_jsonl_rows(path)
    axis = _infer_axis(rows)
    seed = int(shuffle_seed) if shuffle_seed is not None else random.randint(1, 2_000_000_000)
    anon, _ = anonymize_and_shuffle(rows, seed=seed)

    api_key_eff = (api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key_eff:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    base_url_eff = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip()
    model_eff = (judge_model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()

    messages = build_messages(axis, anon)
    content = call_deepseek(
        messages=messages,
        api_key=api_key_eff,
        base_url=base_url_eff,
        model=model_eff,
    )
    scores = parse_scores(content, axis=axis)

    # Build join map from response_number -> {axis, coherence} in original numbering.
    by_num: dict[int, dict[str, float]] = {}
    for obj in scores:
        try:
            rn = int(obj["response_number"])
            ax = float(obj[axis])
            coh = float(obj["coherence"])
        except Exception:
            continue
        by_num[rn] = {axis: ax, "coherence": coh}

    created_at = datetime.now(timezone.utc).isoformat()
    merged: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        rn = i + 1
        s = by_num.get(rn)
        merged.append(
            {
                "response_number": rn,
                "scenario_id": r.get("scenario_id"),
                "multiplier": float(r.get("multiplier")) if r.get("multiplier") is not None else None,
                "llm_axis": s.get(axis) if s else None,
                "llm_coherence": s.get("coherence") if s else None,
            }
        )

    # Write an unscrambled per-response scores JSONL (one line per original row).
    LLM_JUDGE_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8") as sf:
        for i, r in enumerate(rows):
            rn = i + 1
            s = by_num.get(rn)
            sf.write(
                json.dumps(
                    {
                        "response_number": rn,
                        "scenario_id": r.get("scenario_id"),
                        "multiplier": float(r.get("multiplier")) if r.get("multiplier") is not None else None,
                        "prompt": r.get("prompt"),
                        "response_text": r.get("response_text"),
                        "llm_axis": s.get(axis) if s else None,
                        "llm_coherence": s.get("coherence") if s else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    artifact = {
        "source_file": path.name,
        "dimension": axis,
        "shuffle_seed": seed,
        "judge_provider": "deepseek",
        "judge_model": model_eff,
        "created_at": created_at,
        "rows": merged,
        "raw_output": content,
    }
    LLM_JUDGE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "skipped": False,
        "artifact": str(out_path.resolve()),
        "scores": str(scores_path.resolve()),
        "dimension": axis,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DeepSeek LLM judge on a responses JSONL file.")
    ap.add_argument("--input", required=True, help="Basename under sweep/results/responses/ (or absolute path)")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed (optional)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing artifact")
    ap.add_argument("--model", default=None, help="DeepSeek model id (optional; else DEEPSEEK_MODEL)")
    ap.add_argument("--base-url", default=None, help="DeepSeek base url (optional; else DEEPSEEK_BASE_URL)")
    ap.add_argument(
        "--dimension",
        choices=("valence", "arousal", "dominance"),
        default=None,
        help="Optional: require input basename to start with this prefix (e.g. valence_).",
    )
    args = ap.parse_args()

    inp = Path(str(args.input))
    if not inp.is_absolute():
        inp = RESPONSES_DIR / inp
    if args.dimension and not inp.name.startswith(f"{args.dimension}_"):
        raise SystemExit(f"input {inp.name!r} does not match --dimension={args.dimension}")
    res = run_judge_on_file(
        inp,
        shuffle_seed=args.seed,
        overwrite=bool(args.overwrite),
        judge_model=args.model,
        base_url=args.base_url,
    )
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

