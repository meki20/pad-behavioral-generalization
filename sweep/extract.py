"""Extract one steering vector for a PAD dimension and inclusive layer range."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _import_train_steering_vector():
    """
    Import ``train_steering_vector`` from the installed ``steering_vectors`` library.

    The repo also has a ``steering_vectors/`` directory of saved ``.pt`` weights (no
    ``__init__.py``). With cwd on ``sys.path``, that folder can shadow the real package
    and break ``from steering_vectors import train_steering_vector``. Prefer the
    site-packages copy when the shadow is present.
    """
    shadow = ROOT / "steering_vectors"
    if not (shadow.is_dir() and not (shadow / "__init__.py").is_file()):
        from steering_vectors import train_steering_vector as _fn  # noqa: E402

        return _fn

    for name in list(sys.modules):
        if name == "steering_vectors" or name.startswith("steering_vectors."):
            del sys.modules[name]

    site_dirs: list[Path] = []
    try:
        import site as _site

        site_dirs.extend(Path(p) for p in _site.getsitepackages())
        u = getattr(_site, "getusersitepackages", lambda: None)()
        if u:
            site_dirs.append(Path(u))
    except Exception:
        pass

    pushed: str | None = None
    for base in site_dirs:
        init = base / "steering_vectors" / "__init__.py"
        if init.is_file():
            pushed = str(base.resolve())
            sys.path.insert(0, pushed)
            break
    if pushed is None:
        raise ImportError(
            "Could not locate installed steering_vectors package in site-packages; "
            "cannot import train_steering_vector while repo folder steering_vectors/ shadows it."
        )
    try:
        mod = importlib.import_module("steering_vectors.train_steering_vector")
        return mod.train_steering_vector
    finally:
        if pushed is not None and sys.path and sys.path[0] == pushed:
            sys.path.pop(0)


train_steering_vector = _import_train_steering_vector()

from load_model import load_model  # noqa: E402

from sweep.paths import PAIRS_DIR, VECTORS_DIR  # noqa: E402

DEFAULT_NUM_LAYERS = 32


def load_pairs(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"Bad pair entry: {item!r}")
    return out


def extract_steering_vector(
    model,
    tokenizer,
    *,
    dimension: str,
    start: int,
    end: int,
    num_layers: int | None = None,
    pairs_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Train and save one steering vector; ``model`` / ``tokenizer`` already loaded."""
    pairs_path = pairs_path or (PAIRS_DIR / f"{dimension}.json")
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    if end < start:
        raise ValueError("end must be >= start")

    n_layers = num_layers
    if n_layers is None:
        n_layers = int(getattr(model.config, "num_hidden_layers", DEFAULT_NUM_LAYERS))
    last = n_layers - 1
    if start < 0 or end > last:
        raise ValueError(f"Layer range out of bounds for num_layers={n_layers}: valid end <= {last}")

    layers = list(range(start, end + 1))
    pairs = load_pairs(pairs_path)
    range_id = f"{dimension}_{start}_{end}"
    print(f"Training {range_id} on layers {layers[0]}..{layers[-1]} ({len(layers)} layers), {len(pairs)} pairs...")

    vec = train_steering_vector(model, tokenizer, pairs, layers=layers)
    out = out_path or (VECTORS_DIR / f"{range_id}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vec, out)
    print(f"Saved {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train one steering vector (single layer range).")
    p.add_argument("--dimension", choices=("valence", "arousal", "dominance"), required=True)
    p.add_argument("--start", type=int, required=True, help="First decoder layer index (inclusive).")
    p.add_argument("--end", type=int, required=True, help="Last decoder layer index (inclusive).")
    p.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="JSON file of [[pos,neg], ...]. Default: sweep/pairs/<dimension>.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pt path. Default: sweep/results/vectors/<dimension>_<start>_<end>.pt",
    )
    p.add_argument("--num-layers", type=int, default=None, help="Max layer index + 1 (default: from model).")
    args = p.parse_args()

    pairs_path = args.pairs or (PAIRS_DIR / f"{args.dimension}.json")
    out_path = args.out

    model, tokenizer = load_model()
    extract_steering_vector(
        model,
        tokenizer,
        dimension=args.dimension,
        start=args.start,
        end=args.end,
        num_layers=args.num_layers,
        pairs_path=pairs_path,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
