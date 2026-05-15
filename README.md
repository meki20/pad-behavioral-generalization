# PAD steering sweep

Layer-range steering experiments for **PAD** (pleasure–arousal–dominance) control in causal language models. This repository includes trained steering vectors, generation and scoring artifacts, analysis rankings, a CLI toolkit under `sweep/`, and a small FastAPI browser UI.

## Reproducibility bundle

The following are **committed** so results can be inspected without re-running the full GPU sweep:

| Path | Contents |
|------|----------|
| `steering_vectors/*.pt` | Trained steering vectors (`steering_vectors` library format) |
| `sweep/results/responses/` | Generation JSONL per layer range and run |
| `sweep/results/scoring/` | PAD classifier scores per response file |
| `sweep/results/analysis/` | Aggregated rankings (`rankings_*.json`, CSV) |
| `sweep/pairs/` | Contrastive training pairs per dimension |
| `sweep/scenarios.json` | Evaluation prompts |


## Requirements

- Python 3.10+
- CUDA GPU recommended for extract / generate / pipeline
- [steering_vectors](https://pypi.org/project/steering-vectors/) (training and applying vectors)
- PyTorch, Transformers, and PAD classifier modules (`pad_config.py`, `emotion_classifier.py`)

### Install dependencies

From the repository root:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Configure the base causal LM in `load_model.py` (`MODEL_NAME`, quantization). Point scoring at your emotion classifier directory (default `emotion_model_final/`).

For LLM judge runs, set `DEEPSEEK_API_KEY` in a `.env` file (see `sweep/server.py`).

## Quick start (UI)

```bash
python sweep/server.py
```

Open http://127.0.0.1:8765 — use **Results** for rankings tables, layer-range heatmaps, and per-scenario charts. Workflow tabs run extract, pipeline, generate, score, LLM judge, and analyze as background jobs.

## CLI examples

```bash
# One vector
python sweep/extract.py --dimension valence --start 8 --end 14 --num-layers 32

# Full layer grid (plan job count)
python -m sweep.pipeline plan --preset full32 --dimensions valence,arousal,dominance

# Run or resume automated sweep
python -m sweep.pipeline run --preset full32 --state sweep/results/pad_pipeline_state.json

# Single generation file
python sweep/generate.py --vector steering_vectors/valence_8_14.pt \
  --range-id valence_8_14 --dimension valence \
  --multipliers=-0.25,0,0.25

python sweep/score.py --input sweep/results/responses/valence_8_14_RUNID.jsonl
python sweep/analyze.py
```

Always run commands from the **repository root** so `from sweep…` imports resolve.

## Root scripts

| Script | Purpose |
|--------|---------|
| `load_model.py` | Load the configured causal LM and tokenizer |
| `extract_valence_pairs.py` | Legacy helper to build valence pair JSON |
| `extract_arousal_pairs.py` | Legacy helper for arousal pairs |
| `extract_dominance_pairs.py` | Legacy helper for dominance pairs |
| `test_steering.py` | Smoke test: apply one vector and generate |
| `orthogonality.py` | Compare three `.pt` vectors (angles, Gram matrix) |
| `pad_config.py` | GoEmotions labels, PAD mapping, training hyperparameters |
| `emotion_classifier.py` | DistilBERT multi-label classifier → PAD scores |
| `training/train.py` | Retrain classifier on GoEmotions (writes `emotion_model_final/`) |

`pad_combined_test.py` is excluded from the release (see `.gitignore`).

## Sweep package

| Module | Role |
|--------|------|
| `sweep/extract.py` | Train one steering vector for a layer range |
| `sweep/generate.py` | Multiplier sweep over scenarios → JSONL |
| `sweep/score.py` | PAD + safety metrics on responses |
| `sweep/analyze.py` | Build rankings from scored JSONL |
| `sweep/pipeline.py` | Automated extract → generate → score |
| `sweep/llm_judge.py` | Optional DeepSeek judge pass |
| `sweep/server.py` | FastAPI UI + job runner |
| `sweep/grids.py` | `full32` layer-range preset |

## Citation

If you use this code or bundled results in academic work, please cite the accompanying paper (details TBD).

## License

TBD.
