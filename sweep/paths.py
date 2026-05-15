"""Repo root resolution for sweep scripts (run from any cwd)."""
from __future__ import annotations

from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SWEEP_DIR.parent
SCENARIOS_PATH = SWEEP_DIR / "scenarios.json"
VECTORS_DIR = SWEEP_DIR / "results" / "vectors"
RESPONSES_DIR = SWEEP_DIR / "results" / "responses"
SCORING_DIR = SWEEP_DIR / "results" / "scoring"
LLM_JUDGE_DIR = SWEEP_DIR / "results" / "llm_judge"
LLM_JUDGE_ARTIFACTS_DIR = LLM_JUDGE_DIR / "artifacts"
LLM_JUDGE_SCORES_DIR = LLM_JUDGE_DIR / "scores"
ANALYSIS_DIR = SWEEP_DIR / "results" / "analysis"
PAIRS_DIR = SWEEP_DIR / "pairs"
STATIC_DIR = SWEEP_DIR / "static"
LOGS_DIR = SWEEP_DIR / "results" / "logs"

for _p in (
    VECTORS_DIR,
    RESPONSES_DIR,
    SCORING_DIR,
    LLM_JUDGE_DIR,
    LLM_JUDGE_ARTIFACTS_DIR,
    LLM_JUDGE_SCORES_DIR,
    ANALYSIS_DIR,
    LOGS_DIR,
    STATIC_DIR,
    PAIRS_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)
