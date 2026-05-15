"""FastAPI server for sweep vectors, rankings, and static UI."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # noqa: BLE001
    load_dotenv = None

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from sweep.paths import (  # noqa: E402
    ANALYSIS_DIR,
    LLM_JUDGE_ARTIFACTS_DIR,
    LLM_JUDGE_SCORES_DIR,
    LOGS_DIR,
    RESPONSES_DIR,
    SCENARIOS_PATH,
    SCORING_DIR,
    STATIC_DIR,
    VECTORS_DIR,
)

from sweep.llm_judge import run_judge_on_file  # noqa: E402

SWEEP_DIR = ROOT / "sweep"
RESULTS_DIR = SWEEP_DIR / "results"
EXTRACT_SCRIPT = SWEEP_DIR / "extract.py"
GENERATE_SCRIPT = SWEEP_DIR / "generate.py"
SCORE_SCRIPT = SWEEP_DIR / "score.py"
ANALYZE_SCRIPT = SWEEP_DIR / "analyze.py"

_extract_jobs: dict[str, dict] = {}
_extract_lock = threading.Lock()

_runner_jobs: dict[str, dict] = {}
_runner_lock = threading.Lock()

_SAFE_FILE = re.compile(r"^[A-Za-z0-9._-]+$")

if load_dotenv is not None:
    load_dotenv(dotenv_path=str(ROOT / ".env"), override=False)

def _load_rankings_dimension(dim: str) -> dict[str, list]:
    p = ANALYSIS_DIR / f"rankings_{dim}.json"
    if not p.is_file():
        return {"by_scenario": [], "mean": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"by_scenario": [], "mean": []}
    return {
        "by_scenario": list(data.get("by_scenario", [])),
        "mean": list(data.get("mean", [])),
    }


def _scan_response_jsonl_metadata() -> tuple[set[str], set[str], set[str]]:
    range_ids: set[str] = set()
    run_ids: set[str] = set()
    scenario_ids: set[str] = set()
    if not RESPONSES_DIR.is_dir():
        return range_ids, run_ids, scenario_ids
    for path in RESPONSES_DIR.glob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("range_id"):
                        range_ids.add(str(obj["range_id"]))
                    if obj.get("run_id"):
                        run_ids.add(str(obj["run_id"]))
                    if obj.get("scenario_id"):
                        scenario_ids.add(str(obj["scenario_id"]))
        except OSError:
            continue
    return range_ids, run_ids, scenario_ids


def _resolve_generation_jsonl(range_id: str, run_id: str | None) -> Path | None:
    patt = f"{range_id}_*.jsonl"
    cands = [p for p in RESPONSES_DIR.glob(patt) if p.is_file()]
    if not cands:
        return None
    if run_id and run_id.strip():
        rid = run_id.strip()
        if not _SAFE_FILE.match(rid):
            return None
        exact = RESPONSES_DIR / f"{range_id}_{rid}.jsonl"
        if exact.is_file():
            return exact
        for p in cands:
            if p.stem == f"{range_id}_{rid}":
                return p
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _safe_name(name: str, suffix: str) -> str:
    if not name.endswith(suffix) or not _SAFE_FILE.match(name):
        raise HTTPException(400, f"invalid file name: {name!r}")
    return name


def _safe_results_json(basename: str) -> Path:
    """Resolve ``sweep/results/<basename>`` with no path traversal."""
    name = _safe_name(basename, ".json")
    p = (RESULTS_DIR / name).resolve()
    root = RESULTS_DIR.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise HTTPException(400, "invalid results path") from e
    return p


app = FastAPI(title="PAD sweep")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/vectors")
def list_vectors() -> list[dict]:
    if not VECTORS_DIR.is_dir():
        return []
    out = []
    for p in sorted(VECTORS_DIR.glob("*.pt")):
        out.append({"name": p.name, "path": str(p.resolve())})
    return out


@app.get("/api/rankings")
def rankings(dimension: str = Query(..., pattern="^(valence|arousal|dominance)$")) -> dict:
    return _load_rankings_dimension(dimension)


@app.get("/api/rankings_mean")
def rankings_mean(dimension: str = Query(..., pattern="^(valence|arousal|dominance)$")) -> list[dict]:
    return _load_rankings_dimension(dimension)["mean"]


@app.get("/api/response-filters")
def response_filters() -> dict:
    """Distinct range_id / run_id / scenario_id from response JSONL + scenarios.json."""
    range_set, run_set, scen_set = _scan_response_jsonl_metadata()
    static_ids: list[str] = []
    if SCENARIOS_PATH.is_file():
        try:
            data = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                static_ids = [str(s.get("id")) for s in data if isinstance(s, dict) and s.get("id")]
        except (json.JSONDecodeError, OSError):
            pass
    scenario_ids = sorted(scen_set | set(static_ids))
    return {
        "range_ids": sorted(range_set),
        "run_ids": sorted(run_set),
        "scenario_ids": scenario_ids,
    }


@app.get("/api/responses")
def api_responses(
    range_id: str = Query(..., pattern=r"^[\w-]+$"),
    scenario_id: str = Query(..., min_length=1, max_length=120),
    run_id: str | None = Query(default=None, max_length=80),
) -> list[dict]:
    if run_id and run_id.strip():
        rid = run_id.strip()
        if not _SAFE_FILE.match(rid):
            raise HTTPException(400, "invalid run_id")
        gen_path = _resolve_generation_jsonl(range_id, rid)
    else:
        gen_path = _resolve_generation_jsonl(range_id, None)
    if gen_path is None:
        raise HTTPException(404, "no matching responses JSONL for range_id (and run_id if set)")
    scores_path = SCORING_DIR / f"{gen_path.stem}.scores.jsonl"
    judge_scores_path = LLM_JUDGE_SCORES_DIR / f"{gen_path.stem}.llm.scores.jsonl"
    score_by_mult: dict[float, dict] = {}
    if scores_path.is_file():
        try:
            with scores_path.open(encoding="utf-8") as sf:
                for raw in sf:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        srow = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(srow.get("scenario_id")) != str(scenario_id):
                        continue
                    score_by_mult[round(float(srow["multiplier"]), 8)] = srow
        except OSError:
            pass

    llm_by_key: dict[tuple[str, float], dict] = {}
    if judge_scores_path.is_file():
        try:
            with judge_scores_path.open(encoding="utf-8") as jf:
                for raw in jf:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(r, dict):
                        continue
                    sid = str(r.get("scenario_id") or "")
                    if not sid:
                        continue
                    m = r.get("multiplier")
                    if m is None:
                        continue
                    try:
                        mm = round(float(m), 8)
                    except Exception:
                        continue
                    llm_by_key[(sid, mm)] = r
        except Exception:
            llm_by_key = {}

    out: list[dict] = []
    try:
        with gen_path.open(encoding="utf-8") as gf:
            for raw in gf:
                line = raw.strip()
                if not line:
                    continue
                try:
                    grow = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(grow.get("scenario_id")) != str(scenario_id):
                    continue
                m = round(float(grow["multiplier"]), 8)
                srow = score_by_mult.get(m)
                jrow = llm_by_key.get((str(scenario_id), m))
                if srow:
                    out.append(
                        {
                            "multiplier": float(grow["multiplier"]),
                            "response_text": grow.get("response_text", ""),
                            "dimension": grow.get("dimension", ""),
                            "pad_v": srow.get("pad_v"),
                            "pad_a": srow.get("pad_a"),
                            "pad_d": srow.get("pad_d"),
                            "expected_v": srow.get("expected_v"),
                            "expected_a": srow.get("expected_a"),
                            "expected_d": srow.get("expected_d"),
                            "llm_axis": jrow.get("llm_axis") if isinstance(jrow, dict) else None,
                            "llm_coherence": jrow.get("llm_coherence") if isinstance(jrow, dict) else None,
                        }
                    )
                else:
                    out.append(
                        {
                            "multiplier": float(grow["multiplier"]),
                            "response_text": grow.get("response_text", ""),
                            "dimension": grow.get("dimension", ""),
                            "pad_v": None,
                            "pad_a": None,
                            "pad_d": None,
                            "expected_v": None,
                            "expected_a": None,
                            "expected_d": None,
                            "llm_axis": jrow.get("llm_axis") if isinstance(jrow, dict) else None,
                            "llm_coherence": jrow.get("llm_coherence") if isinstance(jrow, dict) else None,
                        }
                    )
    except OSError as e:
        raise HTTPException(500, f"failed to read generations: {e}") from e

    out.sort(key=lambda r: float(r["multiplier"]))
    return out


@app.get("/api/scenarios")
def scenarios() -> list:
    if not SCENARIOS_PATH.is_file():
        raise HTTPException(404, "scenarios.json missing")
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


@app.get("/api/response-files")
def list_response_files() -> list[dict]:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(RESPONSES_DIR.glob("*.jsonl"), reverse=True):
        try:
            st = p.stat()
            out.append(
                {
                    "name": p.name,
                    "path": str(p.resolve()),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                }
            )
        except OSError:
            continue
    return out


class GenerateBody(BaseModel):
    vector: str = Field(..., description="Basename under sweep/results/vectors/, e.g. valence_8_14.pt")
    range_id: str = Field(..., min_length=3, max_length=120, pattern=r"^[\w-]+$")
    dimension: Literal["valence", "arousal", "dominance"] | None = Field(
        default=None,
        description="Inferred from range_id prefix if omitted.",
    )
    scenario_ids: list[str] | None = Field(
        default=None,
        description="If set, only these scenario ids; if null, all scenarios.",
    )
    multipliers: str = Field(
        default="-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25",
        max_length=500,
    )
    run_id: str | None = Field(default=None, max_length=80)
    resume: bool = False
    max_new_tokens: int = Field(100, ge=1, le=1024)
    do_sample: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class ScoreBody(BaseModel):
    input_file: str = Field(..., description="Basename under sweep/results/responses/, e.g. run.jsonl")
    resume: bool = True
    emotion_model: str | None = Field(default=None, max_length=500)


class LLMJudgeBody(BaseModel):
    files: list[str] | None = Field(
        default=None,
        description="Basenames under sweep/results/responses/.jsonl",
    )
    all: bool = False
    dimension: Literal["valence", "arousal", "dominance"] | None = Field(
        default=None,
        description="If set, only score response files whose basename starts with this prefix (e.g. valence_).",
    )
    shuffle_seed: int | None = Field(default=None)
    overwrite: bool = False


class PipelineExportBody(BaseModel):
    preset: Literal["full32"] = "full32"
    output: str = Field(..., max_length=120, description="Basename under sweep/results/, e.g. playlist_full32.json")


class PipelineRunBody(BaseModel):
    state: str = Field(..., max_length=120, description="State JSON basename under sweep/results/")
    preset: Literal["full32"] | None = None
    playlist: str | None = Field(default=None, max_length=120)
    dimensions: str = Field("valence,arousal,dominance", max_length=200)
    force_new_state: bool = False
    run_id: str | None = Field(default=None, max_length=80)
    phase: Literal["all", "extract", "generate", "score"] = "all"
    from_job: int = Field(0, ge=0)
    limit: int | None = Field(default=None, ge=1)
    num_layers: int = Field(32, ge=1, le=128)
    skip_existing_vectors: bool = True
    max_new_tokens: int = Field(100, ge=1, le=1024)
    do_sample: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    cuda_empty_cache: bool = False
    scenario_ids: str | None = Field(default=None, max_length=500)


class PipelineStateBody(BaseModel):
    state: str = Field(..., max_length=120)


@app.get("/api/pipeline/plan")
def pipeline_plan(
    preset: str | None = Query(None),
    playlist: str | None = Query(None),
    dimensions: str = Query("valence,arousal,dominance"),
) -> dict:
    if not preset and not playlist:
        raise HTTPException(400, "Provide query param preset=full32 or playlist=<basename>.json")
    cmd = [sys.executable, "-m", "sweep.pipeline", "plan", "--dimensions", dimensions]
    if preset:
        if preset != "full32":
            raise HTTPException(400, "invalid preset")
        cmd.extend(["--preset", preset])
    else:
        pl = _safe_results_json(playlist or "")
        if not pl.is_file():
            raise HTTPException(400, f"playlist not found: {pl.name}")
        cmd.extend(["--playlist", str(pl.resolve())])
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


@app.get("/api/pipeline/states")
def pipeline_list_states(limit: int = Query(40, ge=1, le=200)) -> list[dict]:
    """JSON files in sweep/results/ (for state / playlist pickers)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if not _SAFE_FILE.match(p.name):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        done = None
        n_jobs = None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data.get("completed_job_indices"), list):
                done = len(data["completed_job_indices"])
            if isinstance(data.get("jobs"), list):
                n_jobs = len(data["jobs"])
        except (json.JSONDecodeError, OSError):
            pass
        out.append(
            {
                "name": p.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "pipeline_jobs": n_jobs,
                "pipeline_completed": done,
            }
        )
        if len(out) >= limit:
            break
    return out


@app.post("/api/pipeline/export")
def pipeline_export(body: PipelineExportBody) -> dict:
    out_path = _safe_results_json(body.output)
    cmd = [
        sys.executable,
        "-m",
        "sweep.pipeline",
        "export",
        "--preset",
        body.preset,
        "-o",
        str(out_path.resolve()),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise HTTPException(
            500,
            detail={"exit_code": proc.returncode, "stderr": proc.stderr, "stdout": proc.stdout},
        )
    return {"output": body.output, "path": str(out_path.resolve()), "stdout": proc.stdout or ""}


@app.post("/api/pipeline/run")
def pipeline_run(body: PipelineRunBody) -> dict:
    state_path = _safe_results_json(body.state)
    need_source = body.force_new_state or not state_path.is_file()
    if need_source and not body.preset and not body.playlist:
        raise HTTPException(400, "preset or playlist required when creating or replacing state")

    if body.run_id is not None:
        rid = body.run_id.strip()
        if rid and not _SAFE_FILE.match(rid):
            raise HTTPException(400, "run_id must contain only letters, digits, ._-")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "sweep.pipeline",
        "run",
        "--state",
        str(state_path.resolve()),
        "--phase",
        body.phase,
        "--dimensions",
        body.dimensions,
        "--num-layers",
        str(body.num_layers),
        "--from-job",
        str(body.from_job),
        f"--max-new-tokens={body.max_new_tokens}",
        f"--temperature={body.temperature}",
        f"--top-p={body.top_p}",
    ]
    if body.preset:
        cmd.extend(["--preset", body.preset])
    if body.playlist:
        pl = _safe_results_json(body.playlist)
        if not pl.is_file():
            raise HTTPException(400, f"playlist not found: {pl.name}")
        cmd.extend(["--playlist", str(pl.resolve())])
    if body.force_new_state:
        cmd.append("--force-new-state")
    if body.run_id and body.run_id.strip():
        cmd.extend(["--run-id", body.run_id.strip()])
    if body.limit is not None:
        cmd.extend(["--limit", str(body.limit)])
    if not body.skip_existing_vectors:
        cmd.append("--no-skip-existing-vectors")
    if body.do_sample:
        cmd.append("--do-sample")
    if body.cuda_empty_cache:
        cmd.append("--cuda-empty-cache")
    if body.scenario_ids and body.scenario_ids.strip():
        cmd.extend(["--scenario-ids", body.scenario_ids.strip()])

    job_id = uuid.uuid4().hex[:12]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"pipeline_{job_id}.log"

    record = {
        "job_id": job_id,
        "kind": "pipeline",
        "status": "running",
        "state_file": body.state,
        "phase": body.phase,
        "log_path": str(log_path.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
    }
    with _runner_lock:
        _runner_jobs[job_id] = record

    def worker() -> None:
        code = -1
        err_msg = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=None,
                    check=False,
                )
                code = int(proc.returncode)
        except Exception as e:  # noqa: BLE001
            err_msg = repr(e)
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n[server] {err_msg}\n")
            except OSError:
                pass
        fin = datetime.now(timezone.utc).isoformat()
        with _runner_lock:
            j = _runner_jobs.get(job_id)
            if j is None:
                return
            j["status"] = "done" if code == 0 else "error"
            j["exit_code"] = code
            j["finished_at"] = fin
            j["error"] = err_msg

    threading.Thread(target=worker, name=f"pipeline-{job_id}", daemon=True).start()
    return {"job_id": job_id, "state": body.state, "log_path": str(log_path.resolve())}


@app.post("/api/pipeline/pause")
def pipeline_pause(body: PipelineStateBody) -> dict:
    state_path = _safe_results_json(body.state)
    pause: Path
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            pause = Path(str(data.get("pause_file", str(state_path.with_suffix(".pause")))))
        except (json.JSONDecodeError, OSError, TypeError):
            pause = state_path.with_suffix(".pause")
    else:
        pause = state_path.with_suffix(".pause")
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.touch()
    return {"ok": True, "pause_file": str(pause.resolve()), "state_existed": state_path.is_file()}


@app.post("/api/pipeline/resume")
def pipeline_resume(body: PipelineStateBody) -> dict:
    state_path = _safe_results_json(body.state)
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            pause = Path(str(data.get("pause_file", str(state_path.with_suffix(".pause")))))
        except (json.JSONDecodeError, OSError, TypeError):
            pause = state_path.with_suffix(".pause")
    else:
        pause = state_path.with_suffix(".pause")
    removed = False
    if pause.is_file():
        pause.unlink()
        removed = True
    return {"ok": True, "pause_file": str(pause.resolve()), "removed": removed}


def _dimension_from_range_id(range_id: str) -> Literal["valence", "arousal", "dominance"] | None:
    m = re.match(r"^(valence|arousal|dominance)_", range_id)
    if not m:
        return None
    return m.group(1)  # type: ignore[return-value]


@app.post("/api/generate")
def start_generate(body: GenerateBody) -> dict:
    vname = _safe_name(body.vector, ".pt")
    vpath = VECTORS_DIR / vname
    if not vpath.is_file():
        raise HTTPException(400, f"vector not found: {vname}")

    dimension = body.dimension or _dimension_from_range_id(body.range_id)
    if dimension is None:
        raise HTTPException(
            400,
            "dimension required or range_id must start with valence_, arousal_, or dominance_",
        )

    if body.run_id is not None:
        rid = body.run_id.strip()
        if not rid:
            raise HTTPException(400, "run_id must be non-empty when provided")
        if not _SAFE_FILE.match(rid):
            raise HTTPException(400, "run_id must contain only letters, digits, ._-")
        run_id = rid
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_jsonl = RESPONSES_DIR / f"{body.range_id}_{run_id}.jsonl"
    job_id = uuid.uuid4().hex[:12]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"generate_{job_id}.log"

    # Values like "-0.25,..." must use --opt=value or argparse treats "-0.25" as a new flag.
    mults = (body.multipliers or "").strip() or "-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25"

    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--vector",
        str(vpath.resolve()),
        "--range-id",
        body.range_id,
        "--dimension",
        dimension,
        f"--multipliers={mults}",
        f"--run-id={run_id}",
        f"--max-new-tokens={body.max_new_tokens}",
        f"--temperature={body.temperature}",
        f"--top-p={body.top_p}",
    ]
    if body.scenario_ids:
        cmd.extend(["--scenario-ids", ",".join(body.scenario_ids)])
    if body.resume:
        cmd.append("--resume")
    if body.do_sample:
        cmd.append("--do-sample")

    record = {
        "job_id": job_id,
        "kind": "generate",
        "status": "running",
        "vector": vname,
        "range_id": body.range_id,
        "dimension": body.dimension,
        "run_id": run_id,
        "output_jsonl": str(out_jsonl.resolve()),
        "log_path": str(log_path.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
    }
    with _runner_lock:
        _runner_jobs[job_id] = record

    def worker() -> None:
        code = -1
        err_msg = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=None,
                    check=False,
                )
                code = int(proc.returncode)
        except Exception as e:  # noqa: BLE001
            err_msg = repr(e)
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n[server] {err_msg}\n")
            except OSError:
                pass
        fin = datetime.now(timezone.utc).isoformat()
        with _runner_lock:
            j = _runner_jobs.get(job_id)
            if j is None:
                return
            j["status"] = "done" if code == 0 else "error"
            j["exit_code"] = code
            j["finished_at"] = fin
            j["error"] = err_msg

    threading.Thread(target=worker, name=f"generate-{job_id}", daemon=True).start()
    return {"job_id": job_id, "run_id": run_id, "output_jsonl": str(out_jsonl.resolve())}


@app.post("/api/score")
def start_score(body: ScoreBody) -> dict:
    fname = _safe_name(body.input_file, ".jsonl")
    in_path = RESPONSES_DIR / fname
    if not in_path.is_file():
        raise HTTPException(400, f"responses file not found: {fname}")

    job_id = uuid.uuid4().hex[:12]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"score_{job_id}.log"

    cmd = [
        sys.executable,
        str(SCORE_SCRIPT),
        "--input",
        str(in_path.resolve()),
    ]
    if body.resume:
        cmd.append("--resume")
    if body.emotion_model:
        emo = Path(body.emotion_model)
        if not emo.is_absolute():
            emo = ROOT / emo
        if not emo.exists():
            raise HTTPException(400, "emotion_model path not found")
        cmd.extend(["--emotion-model", str(emo.resolve())])

    record = {
        "job_id": job_id,
        "kind": "score",
        "status": "running",
        "input_file": fname,
        "log_path": str(log_path.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
    }
    with _runner_lock:
        _runner_jobs[job_id] = record

    def worker() -> None:
        code = -1
        err_msg = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=None,
                    check=False,
                )
                code = int(proc.returncode)
        except Exception as e:  # noqa: BLE001
            err_msg = repr(e)
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n[server] {err_msg}\n")
            except OSError:
                pass
        fin = datetime.now(timezone.utc).isoformat()
        with _runner_lock:
            j = _runner_jobs.get(job_id)
            if j is None:
                return
            j["status"] = "done" if code == 0 else "error"
            j["exit_code"] = code
            j["finished_at"] = fin
            j["error"] = err_msg

    threading.Thread(target=worker, name=f"score-{job_id}", daemon=True).start()
    return {"job_id": job_id, "input": str(in_path.resolve())}


@app.post("/api/llm-judge/run")
def start_llm_judge(body: LLMJudgeBody) -> dict:
    dim = body.dimension
    prefix = f"{dim}_" if dim else None

    files: list[str] = []
    if body.all:
        for p in sorted(RESPONSES_DIR.glob("*.jsonl"), reverse=True):
            if not _SAFE_FILE.match(p.name):
                continue
            if prefix and not p.name.startswith(prefix):
                continue
            files.append(p.name)
    else:
        if not body.files:
            raise HTTPException(400, "provide files[] or set all=true")
        for f in body.files:
            name = _safe_name(str(f), ".jsonl")
            if prefix and not name.startswith(prefix):
                raise HTTPException(
                    400,
                    f"file {name!r} does not match dimension filter {dim!r} (expected basename to start with {prefix!r})",
                )
            files.append(name)

    if not files:
        raise HTTPException(
            400,
            "no matching responses .jsonl files"
            + (f" for dimension={dim}" if dim else ""),
        )

    for f in files:
        p = RESPONSES_DIR / f
        if not p.is_file():
            raise HTTPException(400, f"responses file not found: {f}")

    job_id = uuid.uuid4().hex[:12]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"llm_judge_{job_id}.log"

    record = {
        "job_id": job_id,
        "kind": "llm_judge",
        "status": "running",
        "dimension": dim,
        "files": files,
        "files_total": len(files),
        "files_done": 0,
        "current_file": None,
        "log_path": str(log_path.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
    }
    with _runner_lock:
        _runner_jobs[job_id] = record

    def worker() -> None:
        code = 0
        err_msg = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                for idx, fname in enumerate(files):
                    with _runner_lock:
                        j = _runner_jobs.get(job_id)
                        if j is not None:
                            j["current_file"] = fname
                            j["files_done"] = idx
                    lf.write(f"[llm_judge] ({idx+1}/{len(files)}) {fname}\n")
                    lf.flush()
                    try:
                        res = run_judge_on_file(
                            RESPONSES_DIR / fname,
                            shuffle_seed=body.shuffle_seed,
                            overwrite=bool(body.overwrite),
                        )
                        lf.write(json.dumps(res, ensure_ascii=False) + "\n")
                        lf.flush()
                    except Exception as e:  # noqa: BLE001
                        code = 1
                        lf.write(f"[llm_judge] error {fname}: {repr(e)}\n")
                        lf.flush()
                with _runner_lock:
                    j = _runner_jobs.get(job_id)
                    if j is not None:
                        j["files_done"] = len(files)
                        j["current_file"] = None
        except Exception as e:  # noqa: BLE001
            code = 1
            err_msg = repr(e)
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n[server] {err_msg}\n")
            except OSError:
                pass

        fin = datetime.now(timezone.utc).isoformat()
        with _runner_lock:
            j = _runner_jobs.get(job_id)
            if j is None:
                return
            j["status"] = "done" if code == 0 else "error"
            j["exit_code"] = code
            j["finished_at"] = fin
            j["error"] = err_msg

    threading.Thread(target=worker, name=f"llm-judge-{job_id}", daemon=True).start()
    return {"job_id": job_id, "files_total": len(files)}


@app.get("/api/llm-judge/list")
def llm_judge_list(limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
    LLM_JUDGE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for p in sorted(
        LLM_JUDGE_ARTIFACTS_DIR.glob("*.llm_judge.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        if not _SAFE_FILE.match(p.name):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)})
        if len(out) >= limit:
            break
    return out


@app.get("/api/llm-judge/artifact")
def llm_judge_artifact(name: str = Query(..., max_length=200)) -> dict:
    fname = _safe_name(name, ".json")
    if not fname.endswith(".llm_judge.json"):
        raise HTTPException(400, "invalid artifact name")
    p = (LLM_JUDGE_ARTIFACTS_DIR / fname).resolve()
    root = LLM_JUDGE_ARTIFACTS_DIR.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise HTTPException(400, "invalid artifact path") from e
    if not p.is_file():
        raise HTTPException(404, "artifact not found")
    return json.loads(p.read_text(encoding="utf-8"))

@app.post("/api/analyze")
def start_analyze() -> dict:
    job_id = uuid.uuid4().hex[:12]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"analyze_{job_id}.log"

    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--scoring-dir",
        str(SCORING_DIR.resolve()),
        "--out-dir",
        str(ANALYSIS_DIR.resolve()),
    ]

    record = {
        "job_id": job_id,
        "kind": "analyze",
        "status": "running",
        "log_path": str(log_path.resolve()),
        "analysis_dir": str(ANALYSIS_DIR.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
    }
    with _runner_lock:
        _runner_jobs[job_id] = record

    def worker() -> None:
        code = -1
        err_msg = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=None,
                    check=False,
                )
                code = int(proc.returncode)
        except Exception as e:  # noqa: BLE001
            err_msg = repr(e)
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n[server] {err_msg}\n")
            except OSError:
                pass
        fin = datetime.now(timezone.utc).isoformat()
        with _runner_lock:
            j = _runner_jobs.get(job_id)
            if j is None:
                return
            j["status"] = "done" if code == 0 else "error"
            j["exit_code"] = code
            j["finished_at"] = fin
            j["error"] = err_msg

    threading.Thread(target=worker, name=f"analyze-{job_id}", daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/runner/{job_id}")
def runner_status(job_id: str, tail_lines: int = Query(160, ge=1, le=4000)) -> dict:
    with _runner_lock:
        job = _runner_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown runner job_id")
    log_tail = ""
    lp = Path(job["log_path"])
    if lp.is_file():
        try:
            text = lp.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            log_tail = "\n".join(lines[-tail_lines:])
        except OSError as e:
            log_tail = f"(could not read log: {e})"
    out = {**job, "log_tail": log_tail}
    if job.get("kind") == "generate" and job.get("output_jsonl"):
        outp = Path(job["output_jsonl"])
        out["output_exists"] = outp.is_file()
    return out


@app.get("/api/runner-jobs")
def runner_list_jobs(limit: int = Query(30, ge=1, le=200)) -> list[dict]:
    with _runner_lock:
        items = list(_runner_jobs.values())
    items.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    slim = []
    for j in items[:limit]:
        slim.append(
            {
                "job_id": j["job_id"],
                "kind": j.get("kind"),
                "status": j["status"],
                "exit_code": j["exit_code"],
                "started_at": j["started_at"],
                "finished_at": j["finished_at"],
            }
        )
    return slim


class ExtractBody(BaseModel):
    dimension: Literal["valence", "arousal", "dominance"]
    start: int = Field(..., ge=0, le=127)
    end: int = Field(..., ge=0, le=127)
    num_layers: int = Field(32, ge=1, le=128)


def _validate_layer_range(start: int, end: int, num_layers: int) -> None:
    last = num_layers - 1
    if end < start:
        raise HTTPException(400, "end must be >= start")
    if start > last or end > last:
        raise HTTPException(400, f"layer indices must be within 0..{last} for num_layers={num_layers}")


@app.post("/api/extract")
def start_extract(body: ExtractBody) -> dict:
    _validate_layer_range(body.start, body.end, body.num_layers)
    job_id = uuid.uuid4().hex[:12]
    range_id = f"{body.dimension}_{body.start}_{body.end}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"extract_{job_id}.log"

    record = {
        "job_id": job_id,
        "status": "running",
        "dimension": body.dimension,
        "start": body.start,
        "end": body.end,
        "num_layers": body.num_layers,
        "range_id": range_id,
        "log_path": str(log_path.resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
        "error": None,
    }
    with _extract_lock:
        _extract_jobs[job_id] = record

    def worker() -> None:
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--dimension",
            body.dimension,
            "--start",
            str(body.start),
            "--end",
            str(body.end),
            "--num-layers",
            str(body.num_layers),
        ]
        code = -1
        err_msg = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=None,
                    check=False,
                )
                code = int(proc.returncode)
        except Exception as e:  # noqa: BLE001
            err_msg = repr(e)
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(f"\n[server] {err_msg}\n")
            except OSError:
                pass
        fin = datetime.now(timezone.utc).isoformat()
        with _extract_lock:
            j = _extract_jobs.get(job_id)
            if j is None:
                return
            j["status"] = "done" if code == 0 else "error"
            j["exit_code"] = code
            j["finished_at"] = fin
            j["error"] = err_msg

    threading.Thread(target=worker, name=f"extract-{job_id}", daemon=True).start()
    return {"job_id": job_id, "range_id": range_id, "log_path": str(log_path.resolve())}


@app.get("/api/extract/{job_id}")
def extract_status(job_id: str, tail_lines: int = Query(120, ge=1, le=2000)) -> dict:
    with _extract_lock:
        job = _extract_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    log_tail = ""
    lp = Path(job["log_path"])
    if lp.is_file():
        try:
            text = lp.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            log_tail = "\n".join(lines[-tail_lines:])
        except OSError as e:
            log_tail = f"(could not read log: {e})"
    out = {**job, "log_tail": log_tail}
    out_vector = VECTORS_DIR / f"{job['range_id']}.pt"
    out["output_exists"] = out_vector.is_file()
    out["output_path"] = str(out_vector.resolve()) if out_vector.is_file() else str(out_vector.resolve())
    return out


@app.get("/api/extract-jobs")
def extract_list_jobs(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    with _extract_lock:
        items = list(_extract_jobs.values())
    items.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    slim = []
    for j in items[:limit]:
        slim.append(
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "range_id": j["range_id"],
                "exit_code": j["exit_code"],
                "started_at": j["started_at"],
                "finished_at": j["finished_at"],
            }
        )
    return slim


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
