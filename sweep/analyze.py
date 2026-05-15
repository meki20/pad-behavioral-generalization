"""Aggregate scoring JSONL; write rankings CSV + JSON (no SQLite)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep.metrics import linear_trend, norm_ols_slope, norm_rmse  # noqa: E402
from sweep.paths import ANALYSIS_DIR, LLM_JUDGE_SCORES_DIR, SCORING_DIR  # noqa: E402

_DIM_INDEX = {"valence": 0, "arousal": 1, "dominance": 2}


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return None
    m = float(np.corrcoef(x, y)[0, 1])
    return m if np.isfinite(m) else None


def _norm_pad_acc(mean_mse: float) -> float:
    return float(1.0 / (1.0 + mean_mse))


def _norm_self(mean_self: float) -> float:
    return float(np.clip(mean_self * 5.0, 0.0, 1.0))


def _norm_range(var_pad: float) -> float:
    return float(np.clip(var_pad * 10.0, 0.0, 1.0))


def _danger_from_row(row: dict) -> int:
    """Integer danger hits from a scoring JSONL row (new or legacy keys)."""
    if "danger_score" in row:
        try:
            return int(row["danger_score"])
        except (TypeError, ValueError):
            return 0
    flags = row.get("danger_hits") or row.get("safety_flags")
    if isinstance(flags, dict) and flags:
        return int(sum(int(v) for v in flags.values()))
    return 0


def _composite(pad_acc: float, coh: float, self_d: float, emo_r: float) -> float:
    """PAD / coherence composite; danger is reported separately and not included."""
    coh_c = float(np.clip(coh, 0.0, 1.0))
    # Original relative weights among four terms; renormalized after dropping safety.
    return (
        (0.30 / 0.9) * pad_acc
        + (0.25 / 0.9) * coh_c
        + (0.20 / 0.9) * self_d
        + (0.15 / 0.9) * emo_r
    )


def _load_scoring_rows(scoring_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(scoring_dir.glob("*.scores.jsonl")):
        try:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return rows


def _parse_range_run_from_stem(stem: str) -> tuple[str, str, str] | None:
    """
    Parse e.g. dominance_12_17_20260512T103855Z -> (range_id, run_id, dim)
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    dim = parts[0]
    if dim not in _DIM_INDEX:
        return None
    range_id = "_".join(parts[:3])
    run_id = "_".join(parts[3:])
    if not run_id:
        return None
    return range_id, run_id, dim


def _load_llm_metrics(llm_scores_dir: Path) -> dict[tuple[str, str, str, str], dict[str, float | None]]:
    """
    Returns mapping (range_id, run_id, dim, scenario_id) -> {"llm_corr": r, "llm_coh": mean}
    """
    out: dict[tuple[str, str, str, str], dict[str, float | None]] = {}
    for path in sorted(llm_scores_dir.glob("*.llm.scores.jsonl")):
        pr = _parse_range_run_from_stem(path.stem.replace(".llm.scores", ""))
        if pr is None:
            continue
        range_id, run_id, dim = pr
        scen_m: dict[str, list[float]] = defaultdict(list)
        scen_v: dict[str, list[float]] = defaultdict(list)
        scen_c: dict[str, list[float]] = defaultdict(list)
        try:
            with path.open(encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = str(r.get("scenario_id") or "")
                    if not sid:
                        continue
                    try:
                        m = float(r.get("multiplier"))
                        v = float(r.get("llm_axis"))
                    except (TypeError, ValueError):
                        continue
                    c = r.get("llm_coherence")
                    if c is not None:
                        try:
                            scen_c[sid].append(float(c))
                        except (TypeError, ValueError):
                            pass
                    scen_m[sid].append(m)
                    scen_v[sid].append(v)
        except OSError:
            continue

        for sid in scen_m.keys():
            m = np.asarray(scen_m[sid], dtype=np.float64)
            v = np.asarray(scen_v[sid], dtype=np.float64)
            corr = _pearson(m, v)
            coh = float(np.mean(scen_c[sid])) if sid in scen_c and scen_c[sid] else None
            out[(range_id, run_id, dim, sid)] = {"llm_corr": corr, "llm_coh": coh}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank layer ranges from sweep/results/scoring/*.scores.jsonl.")
    ap.add_argument("--scoring-dir", type=Path, default=SCORING_DIR)
    ap.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = ap.parse_args()

    scoring_dir = args.scoring_dir
    if not scoring_dir.is_absolute():
        scoring_dir = ROOT / scoring_dir
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_scoring_rows(scoring_dir)
    llm_map = _load_llm_metrics(LLM_JUDGE_SCORES_DIR)
    groups: dict[tuple[str, str, str, str], dict[str, list]] = defaultdict(
        lambda: {
            "m": [],
            "pad_dim": [],
            "mse": [],
            "coh": [],
            "self": [],
            "danger": [],
        }
    )

    for r in raw_rows:
        range_id = str(r.get("range_id", ""))
        run_id = str(r.get("run_id", ""))
        dim = str(r.get("dimension", ""))
        scenario_id = str(r.get("scenario_id", ""))
        if not range_id or not run_id or dim not in _DIM_INDEX or not scenario_id:
            continue
        idx = _DIM_INDEX[dim]
        pad = (float(r["pad_v"]), float(r["pad_a"]), float(r["pad_d"]))[idx]
        key = (range_id, run_id, dim, scenario_id)
        g = groups[key]
        g["m"].append(float(r["multiplier"]))
        g["pad_dim"].append(float(pad))
        g["mse"].append(float(r.get("mse_active", 0.0)))
        g["coh"].append(float(r.get("coherence", 0.0)))
        g["self"].append(float(r.get("self_ratio", 0.0)))
        g["danger"].append(_danger_from_row(r))

    mean_components: dict[tuple[str, str, str], dict[str, list]] = defaultdict(
        lambda: {
            "pad_acc": [],
            "coh": [],
            "self": [],
            "emo": [],
            "danger": [],
            "n_ols": [],
            "n_rmse": [],
            "llm_corr": [],
            "llm_coh": [],
        }
    )

    by_dim: dict[str, dict[str, list]] = {
        "valence": {"by_scenario": [], "mean": []},
        "arousal": {"by_scenario": [], "mean": []},
        "dominance": {"by_scenario": [], "mean": []},
    }

    for (range_id, run_id, dim, scenario_id), g in groups.items():
        m = np.asarray(g["m"], dtype=np.float64)
        order = np.argsort(m)
        m = m[order]
        pad_dim = np.asarray(g["pad_dim"], dtype=np.float64)[order]

        mean_mse = float(np.mean(g["mse"])) if g["mse"] else 0.0
        mean_coh = float(np.mean(g["coh"])) if g["coh"] else 0.0
        mean_self = float(np.mean(g["self"])) if g["self"] else 0.0
        danger_sum = int(sum(int(x) for x in g["danger"])) if g["danger"] else 0
        var_pad = float(np.var(pad_dim)) if pad_dim.size else 0.0

        slope, _intercept, rmse_fit, mad_fit = linear_trend(m, pad_dim)
        pr = _pearson(m, pad_dim)
        n_ols = norm_ols_slope(slope)
        n_rmse = norm_rmse(rmse_fit)
        pad_acc = _norm_pad_acc(mean_mse)
        self_n = _norm_self(mean_self)
        emo_n = _norm_range(var_pad)
        comp = _composite(pad_acc, mean_coh, self_n, emo_n)

        llm = llm_map.get((range_id, run_id, dim, scenario_id))
        llm_corr = llm.get("llm_corr") if llm else None
        llm_coh = llm.get("llm_coh") if llm else None

        row_out = {
            "run_id": run_id,
            "range_id": range_id,
            "scenario_id": scenario_id,
            "composite": comp,
            "pad_accuracy": pad_acc,
            "ols_slope": float(slope),
            "rmse_fit": float(rmse_fit),
            "mad_fit": float(mad_fit),
            "norm_ols": float(n_ols),
            "norm_rmse": float(n_rmse),
            "coherence": mean_coh,
            "self_direction": self_n,
            "emotional_range": emo_n,
            "danger_score": danger_sum,
            "pearson_r": pr,
            "llm_corr": llm_corr,
            "llm_coherence": llm_coh,
            "n": int(m.size),
        }
        by_dim[dim]["by_scenario"].append(row_out)

        mk = (range_id, run_id, dim)
        mc = mean_components[mk]
        mc["pad_acc"].append(pad_acc)
        mc["coh"].append(mean_coh)
        mc["self"].append(self_n)
        mc["emo"].append(emo_n)
        mc["danger"].append(danger_sum)
        mc["n_ols"].append(n_ols)
        mc["n_rmse"].append(n_rmse)
        if llm_corr is not None and np.isfinite(llm_corr):
            mc["llm_corr"].append(float(llm_corr))
        if llm_coh is not None and np.isfinite(llm_coh):
            mc["llm_coh"].append(float(llm_coh))

    for (range_id, run_id, dim), mc in mean_components.items():
        pad_acc_m = float(np.mean(mc["pad_acc"]))
        coh_m = float(np.mean(mc["coh"]))
        self_m = float(np.mean(mc["self"]))
        emo_m = float(np.mean(mc["emo"]))
        comp_m = _composite(pad_acc_m, coh_m, self_m, emo_m)
        n_ols_m = float(np.mean(mc["n_ols"]))
        n_rmse_m = float(np.mean(mc["n_rmse"]))
        n_scen = len(mc["pad_acc"])
        danger_m = int(round(float(np.mean(mc["danger"])))) if mc["danger"] else 0
        llm_corr_m = float(np.mean(mc["llm_corr"])) if mc["llm_corr"] else None
        llm_coh_m = float(np.mean(mc["llm_coh"])) if mc["llm_coh"] else None
        by_dim[dim]["mean"].append(
            {
                "run_id": run_id,
                "range_id": range_id,
                "scenario_id": "__mean__",
                "composite": comp_m,
                "pad_accuracy": pad_acc_m,
                "ols_slope": None,
                "rmse_fit": None,
                "mad_fit": None,
                "norm_ols": float(n_ols_m),
                "norm_rmse": float(n_rmse_m),
                "coherence": coh_m,
                "self_direction": self_m,
                "emotional_range": emo_m,
                "danger_score": danger_m,
                "pearson_r": None,
                "llm_corr": llm_corr_m,
                "llm_coherence": llm_coh_m,
                "n": n_scen,
            }
        )

    gen_iso = datetime.now(timezone.utc).isoformat()
    for dim in ("valence", "arousal", "dominance"):
        payload = {
            "dimension": dim,
            "generated_at": gen_iso,
            "by_scenario": sorted(
                by_dim[dim]["by_scenario"],
                key=lambda r: (-float(r["composite"]), r["range_id"], r["run_id"], r["scenario_id"]),
            ),
            "mean": sorted(
                by_dim[dim]["mean"],
                key=lambda r: (-float(r["composite"]), r["range_id"], r["run_id"]),
            ),
        }
        jpath = out_dir / f"rankings_{dim}.json"
        jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {jpath}")

        csv_path = out_dir / f"rankings_{dim}.csv"
        mean_rows = payload["mean"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "run_id",
                    "range_id",
                    "composite",
                    "pad_accuracy",
                    "coherence",
                    "llm_corr",
                    "llm_coherence",
                    "self_direction",
                    "emotional_range",
                    "danger_score",
                    "n_scenarios",
                ]
            )
            for row in mean_rows:
                w.writerow(
                    [
                        row["run_id"],
                        row["range_id"],
                        row["composite"],
                        row["pad_accuracy"],
                        row["coherence"],
                        row.get("llm_corr"),
                        row.get("llm_coherence"),
                        row["self_direction"],
                        row["emotional_range"],
                        row["danger_score"],
                        row["n"],
                    ]
                )
        print(f"Wrote {csv_path} ({len(mean_rows)} rows)")


if __name__ == "__main__":
    main()
