"""Automated extract → generate → score sweep with preset grids, pause file, and resumable state.

Examples:

  # Inspect job count (192 jobs for full32 × three dimensions)
  python -m sweep.pipeline plan --preset full32

  # Export editable playlist JSON (same grid / multipliers)
  python -m sweep.pipeline export --preset full32 -o sweep/results/my_playlist.json

  # Start or resume a run (state file stores run_id, job list, completed indices)
  python -m sweep.pipeline run --preset full32 --state sweep/results/pad_run_state.json

  # Pause between jobs: create pause file, then remove when ready
  python -m sweep.pipeline pause --state sweep/results/pad_run_state.json
  python -m sweep.pipeline resume --state sweep/results/pad_run_state.json

  # Staged runs (same state path)
  python -m sweep.pipeline run --state sweep/results/pad_run_state.json --phase extract
  python -m sweep.pipeline run --state sweep/results/pad_run_state.json --phase generate
  python -m sweep.pipeline run --state sweep/results/pad_run_state.json --phase score

  # Valence-only, first 3 layer ranges
  python -m sweep.pipeline run --preset full32 --dimensions valence --state ... --limit 3

Phases ``all`` / ``extract`` / ``generate`` load the causal LM once per invocation; ``score``
loads the emotion classifier once. JSONL outputs use ``{range_id}_{run_id}.jsonl`` with
``--resume`` semantics; scoring writes ``sweep/results/scoring/{stem}.scores.jsonl``. A job
index is appended to ``completed_job_indices`` only after its JSONL is scored in ``--phase all``.

For ``--phase extract``, ``--phase generate``, or ``--phase score`` alone, indices in
``completed_job_indices`` are **ignored** so you can revisit every vector job (e.g.\
after pre-extracting all .pt files, run generate-only across the full playlist; row-level
resume still avoids duplicate (scenario, multiplier) rows in JSONL).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # Append so ``from steering_vectors import …`` resolves to the installed library, not
    # the repo folder ``steering_vectors/`` (saved .pt files only).
    sys.path.append(str(ROOT))

from load_model import load_model  # noqa: E402

from sweep.paths import RESPONSES_DIR, SCENARIOS_PATH, VECTORS_DIR  # noqa: E402
from sweep.extract import extract_steering_vector  # noqa: E402
from sweep.generate import load_scenarios, resolve_system_prompt, run_generation_job  # noqa: E402
from sweep.grids import (  # noqa: E402
    FULL32_STARTS,
    FULL32_WIDTHS,
    MULTIPLIERS_AROUSAL,
    MULTIPLIERS_DOMINANCE,
    MULTIPLIERS_VALENCE,
    build_full32_jobs,
    iter_layer_ranges,
)
from sweep.score import score_jsonl_rows  # noqa: E402

from emotion_classifier import EmotionClassifier  # noqa: E402

Phase = Literal["all", "extract", "generate", "score"]

STATE_VERSION = 1


def _wait_if_paused(pause_file: Path, poll_s: float = 2.0) -> None:
    while pause_file.is_file():
        print(f"[pipeline] paused ({pause_file}); remove file to continue...")
        time.sleep(poll_s)


def _default_multipliers() -> dict[str, list[float]]:
    return {
        "valence": list(MULTIPLIERS_VALENCE),
        "arousal": list(MULTIPLIERS_AROUSAL),
        "dominance": list(MULTIPLIERS_DOMINANCE),
    }


def expand_playlist_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build job dicts: dimension, start, end, multipliers (list[float])."""
    ver = data.get("version", 1)
    if ver != 1:
        raise ValueError(f"Unsupported playlist version: {ver}")

    if "jobs" in data and isinstance(data["jobs"], list):
        out = []
        for j in data["jobs"]:
            out.append(
                {
                    "dimension": str(j["dimension"]),
                    "start": int(j["start"]),
                    "end": int(j["end"]),
                    "multipliers": [float(x) for x in j["multipliers"]],
                }
            )
        return out

    num_layers = int(data.get("num_layers", 32))
    last_layer = int(data.get("last_layer", num_layers - 1))
    starts = tuple(int(x) for x in data["starts"])
    widths = tuple(int(x) for x in data["widths"])
    dims = tuple(str(d) for d in data["dimensions"])
    mults = data.get("multipliers") or _default_multipliers()
    ranges = iter_layer_ranges(starts=starts, widths=widths, last_layer=last_layer)
    jobs: list[dict[str, Any]] = []
    for dim in dims:
        if dim not in mults:
            raise KeyError(f"multipliers missing for dimension {dim!r}")
        mlist = [float(x) for x in mults[dim]]
        for start, end in ranges:
            jobs.append(
                {
                    "dimension": dim,
                    "start": start,
                    "end": end,
                    "multipliers": mlist,
                }
            )
    return jobs


def jobs_from_preset(preset: str, dimensions: tuple[str, ...]) -> list[dict[str, Any]]:
    if preset != "full32":
        raise ValueError(f"Unknown preset {preset!r} (try: full32)")
    for d in dimensions:
        if d not in ("valence", "arousal", "dominance"):
            raise ValueError(f"Invalid dimension {d!r}")
    typed: tuple[Any, ...] = tuple(dimensions)  # type: ignore[assignment]
    built = build_full32_jobs(typed)  # type: ignore[arg-type]
    return [
        {
            "dimension": j.dimension,
            "start": j.start,
            "end": j.end,
            "multipliers": list(j.multipliers()),
        }
        for j in built
    ]


def export_playlist_template(path: Path, *, preset: str) -> None:
    if preset != "full32":
        raise SystemExit(f"Unknown preset {preset!r}")
    doc = {
        "version": 1,
        "description": "32-layer model: starts × widths with end <= last_layer; per-dimension multipliers.",
        "num_layers": 32,
        "last_layer": 31,
        "starts": list(FULL32_STARTS),
        "widths": list(FULL32_WIDTHS),
        "dimensions": ["valence", "arousal", "dominance"],
        "multipliers": _default_multipliers(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    n = len(expand_playlist_dict(doc))
    print(f"Wrote {path} ({n} jobs when expanded)")


def _vector_path(job: dict[str, Any]) -> Path:
    return VECTORS_DIR / f"{job['dimension']}_{job['start']}_{job['end']}.pt"


def _jsonl_path(job: dict[str, Any], run_id: str) -> Path:
    rid = f"{job['dimension']}_{job['start']}_{job['end']}"
    return RESPONSES_DIR / f"{rid}_{run_id}.jsonl"


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_plan(args: argparse.Namespace) -> None:
    if not args.preset and not args.playlist:
        raise SystemExit("plan requires --preset or --playlist")
    if args.preset:
        dims = tuple(d.strip() for d in args.dimensions.split(",") if d.strip())
        jobs = jobs_from_preset(args.preset, dims)
    else:
        data = json.loads(Path(args.playlist).read_text(encoding="utf-8"))
        jobs = expand_playlist_dict(data)
    print(f"jobs: {len(jobs)}")
    for i, j in enumerate(jobs[:12]):
        print(f"  {i:4d}  {j['dimension']}_{j['start']}_{j['end']}  m={len(j['multipliers'])} mults")
    if len(jobs) > 12:
        print(f"  ... ({len(jobs) - 12} more)")


def cmd_run(args: argparse.Namespace) -> None:
    state_path = Path(args.state)

    jobs_new: list[dict[str, Any]] | None = None
    if args.preset:
        dims = tuple(d.strip() for d in args.dimensions.split(",") if d.strip())
        jobs_new = jobs_from_preset(args.preset, dims)
    elif args.playlist:
        data = json.loads(Path(args.playlist).read_text(encoding="utf-8"))
        jobs_new = expand_playlist_dict(data)

    if state_path.is_file() and not args.force_new_state:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            raise SystemExit(f"Unsupported state version in {state_path}")
        if jobs_new is not None and jobs_new != state["jobs"]:
            print(
                "[pipeline] warning: --preset/--playlist job list differs from state file; "
                "using jobs from state (resume). Use --force-new-state to replace.",
                file=sys.stderr,
            )
    else:
        if jobs_new is None:
            raise SystemExit(
                "Creating state requires --preset or --playlist "
                "(or resume with an existing --state file without --force-new-state)."
            )
        pause_default = state_path.with_suffix(".pause")
        pause_p = Path(args.pause_file) if args.pause_file else pause_default
        run_new = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        state = {
            "version": STATE_VERSION,
            "run_id": run_new,
            "jobs": jobs_new,
            "completed_job_indices": [],
            "pause_file": str(pause_p),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[pipeline] wrote new state {state_path} run_id={run_new} jobs={len(jobs_new)}")

    jobs = state["jobs"]
    run_id = state["run_id"]
    pause_path = Path(state["pause_file"])

    completed = set(int(x) for x in state.get("completed_job_indices", []))
    scenarios = load_scenarios(Path(args.scenarios))
    ids_filter = None
    if args.scenario_ids:
        ids_filter = {s.strip() for s in args.scenario_ids.split(",") if s.strip()}

    lim_start = int(args.from_job)
    lim_end = len(jobs) if args.limit is None else min(len(jobs), lim_start + int(args.limit))

    if lim_start >= lim_end:
        print(f"[pipeline] empty job slice [{lim_start}, {lim_end}); nothing to do.")
        return

    phase: Phase = args.phase  # type: ignore[assignment]
    skip_ext = bool(args.skip_existing_vectors)
    sys_prompt = resolve_system_prompt(Path(args.system_prompt_file)) if args.system_prompt_file else None

    model = None
    tokenizer = None
    try:
        if phase in ("all", "extract", "generate"):
            print("[pipeline] loading causal LM...")
            model, tokenizer = load_model()
            try:
                if phase in ("all", "extract"):
                    for idx in range(lim_start, lim_end):
                        if phase == "all" and idx in completed:
                            continue
                        _wait_if_paused(pause_path)
                        job = jobs[idx]
                        vpath = _vector_path(job)
                        if skip_ext and vpath.is_file():
                            print(f"[pipeline] [{idx}] skip extract (exists): {vpath.name}")
                        else:
                            print(f"[pipeline] [{idx}] extract {job['dimension']}_{job['start']}_{job['end']}")
                            extract_steering_vector(
                                model,
                                tokenizer,
                                dimension=job["dimension"],
                                start=job["start"],
                                end=job["end"],
                                num_layers=args.num_layers,
                                out_path=vpath,
                            )

                if phase in ("all", "generate"):
                    for idx in range(lim_start, lim_end):
                        # In generate-only mode, ignore completed_job_indices so every
                        # vector job runs (row-level --resume in JSONL still applies).
                        if phase == "all" and idx in completed:
                            continue
                        _wait_if_paused(pause_path)
                        job = jobs[idx]
                        vpath = _vector_path(job)
                        if not vpath.is_file():
                            raise FileNotFoundError(f"missing vector for job {idx}: {vpath}")
                        vec = torch.load(vpath, map_location="cpu", weights_only=False)
                        out_jsonl = _jsonl_path(job, run_id)
                        print(f"[pipeline] [{idx}] generate -> {out_jsonl.name}")
                        run_generation_job(
                            model,
                            tokenizer,
                            vec,
                            range_id=f"{job['dimension']}_{job['start']}_{job['end']}",
                            dimension=job["dimension"],
                            scenarios=scenarios,
                            scenario_ids_filter=ids_filter,
                            multipliers=job["multipliers"],
                            out_path=out_jsonl,
                            run_id=run_id,
                            resume=True,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            system_prompt=sys_prompt,
                        )
            finally:
                if model is not None:
                    del model, tokenizer
                    model = tokenizer = None
                if args.cuda_empty_cache:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        if phase in ("all", "score"):
            emo_path = Path(args.emotion_model)
            if not emo_path.is_absolute():
                emo_path = ROOT / emo_path
            print("[pipeline] loading emotion classifier...")
            clf = EmotionClassifier(str(emo_path))
            for idx in range(lim_start, lim_end):
                if phase == "all" and idx in completed:
                    continue
                _wait_if_paused(pause_path)
                job = jobs[idx]
                out_jsonl = _jsonl_path(job, run_id)
                if not out_jsonl.is_file():
                    print(f"[pipeline] [{idx}] skip score (no JSONL yet): {out_jsonl.name}")
                    continue
                print(f"[pipeline] [{idx}] score <- {out_jsonl.name}")
                score_jsonl_rows(clf, out_jsonl, resume=True)
                completed.add(idx)
                state["completed_job_indices"] = sorted(completed)
                _save_state(state_path, state)

    except KeyboardInterrupt:
        print("\n[pipeline] interrupted; state saved — re-run with same --state to resume.")
        _save_state(state_path, {**state, "completed_job_indices": sorted(completed)})
        raise SystemExit(130) from None

    print(f"[pipeline] done. completed {len(completed)}/{len(jobs)} jobs. run_id={run_id}")


def cmd_pause(args: argparse.Namespace) -> None:
    st = json.loads(Path(args.state).read_text(encoding="utf-8"))
    pause = Path(st.get("pause_file", Path(args.state).with_suffix(".pause")))
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.touch()
    print(f"Created pause file: {pause}")


def cmd_resume(args: argparse.Namespace) -> None:
    st = json.loads(Path(args.state).read_text(encoding="utf-8"))
    pause = Path(st.get("pause_file", Path(args.state).with_suffix(".pause")))
    if pause.is_file():
        pause.unlink()
        print(f"Removed pause file: {pause}")
    else:
        print("(no pause file)")


def main() -> None:
    ap = argparse.ArgumentParser(description="PAD sweep pipeline: playlist, extract, generate, score.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Print job count and sample rows.")
    p_plan.add_argument("--preset", type=str, default=None, choices=("full32",))
    p_plan.add_argument("--playlist", type=str, default=None, help="Playlist JSON (expanded or compact).")
    p_plan.add_argument(
        "--dimensions",
        type=str,
        default="valence,arousal,dominance",
        help="Comma dims when using --preset.",
    )
    p_plan.set_defaults(func=cmd_plan)

    p_exp = sub.add_parser("export", help="Write example playlist JSON.")
    p_exp.add_argument("--preset", type=str, required=True, choices=("full32",))
    p_exp.add_argument("-o", "--output", type=Path, required=True)

    def cmd_export(a: argparse.Namespace) -> None:
        export_playlist_template(a.output, preset=a.preset)

    p_exp.set_defaults(func=cmd_export)

    p_run = sub.add_parser("run", help="Run or resume pipeline using state file.")
    p_run.add_argument("--preset", type=str, default=None, choices=("full32",))
    p_run.add_argument("--playlist", type=str, default=None)
    p_run.add_argument(
        "--state",
        type=str,
        required=True,
        help="JSON path for run_id, job list, completed indices, pause_file path.",
    )
    p_run.add_argument(
        "--dimensions",
        type=str,
        default="valence,arousal,dominance",
        help="Comma dims when using --preset (ignored when resuming from state).",
    )
    p_run.add_argument("--run-id", type=str, default=None, help="Fixed run_id when creating new state.")
    p_run.add_argument(
        "--force-new-state",
        action="store_true",
        help="Ignore existing state file and create a fresh run_id + job list.",
    )
    p_run.add_argument(
        "--pause-file",
        type=str,
        default=None,
        help="Override pause file path stored in state (only when initializing).",
    )
    p_run.add_argument(
        "--phase",
        choices=("all", "extract", "generate", "score"),
        default="all",
        help="Run only one stage (score loads classifier only).",
    )
    p_run.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    p_run.add_argument("--scenario-ids", type=str, default=None)
    p_run.add_argument("--num-layers", type=int, default=32)
    p_run.add_argument("--skip-existing-vectors", action="store_true", default=True)
    p_run.add_argument("--no-skip-existing-vectors", action="store_false", dest="skip_existing_vectors")
    p_run.add_argument("--from-job", type=int, default=0)
    p_run.add_argument("--limit", type=int, default=None, help="Max jobs from --from-job.")
    p_run.add_argument("--max-new-tokens", type=int, default=100)
    p_run.add_argument("--do-sample", action="store_true")
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--top-p", type=float, default=0.9)
    p_run.add_argument("--system-prompt-file", type=Path, default=None)
    p_run.add_argument("--emotion-model", type=Path, default=ROOT / "emotion_model_final")
    p_run.add_argument("--cuda-empty-cache", action="store_true", help="After LM phase, call torch.cuda.empty_cache().")
    p_run.set_defaults(func=cmd_run)

    p_pause = sub.add_parser("pause", help="Create pause file from state.")
    p_pause.add_argument("--state", type=str, required=True)
    p_pause.set_defaults(func=cmd_pause)

    p_res = sub.add_parser("resume", help="Remove pause file from state.")
    p_res.add_argument("--state", type=str, required=True)
    p_res.set_defaults(func=cmd_resume)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
