"""Print or run many extract.py invocations for the layer sweep (32 layers by default).

For a full extract → generate → score playlist with shared multipliers and resumable
state, prefer ``python -m sweep.pipeline`` (see ``sweep/pipeline.py``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACT = Path(__file__).resolve().parent / "extract.py"
DEFAULT_NUM_LAYERS = 32


def iter_ranges(
    *,
    num_layers: int,
    start_min: int,
    start_max_inclusive: int,
    width_min: int,
    width_max_inclusive: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for start in range(start_min, start_max_inclusive + 1):
        for end in range(start + width_min, min(start + width_max_inclusive + 1, num_layers)):
            ranges.append((start, end))
    return ranges


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch layer-range sweep helper.")
    ap.add_argument("--dimension", choices=("valence", "arousal", "dominance"), default=None)
    ap.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="Total decoder layers (default 32).")
    ap.add_argument("--start-min", type=int, default=8)
    ap.add_argument("--start-max", type=int, default=28)
    ap.add_argument("--width-min", type=int, default=2, help="end - start, minimum (inclusive span).")
    ap.add_argument("--width-max", type=int, default=8, help="Maximum inclusive width (end - start).")
    ap.add_argument("--execute", action="store_true", help="Run python sweep/extract.py for each combo.")
    args = ap.parse_args()

    combos = []
    dims = [args.dimension] if args.dimension else ["valence", "arousal", "dominance"]
    for dim in dims:
        for start, end in iter_ranges(
            num_layers=args.num_layers,
            start_min=args.start_min,
            start_max_inclusive=args.start_max,
            width_min=args.width_min,
            width_max_inclusive=args.width_max,
        ):
            combos.append((dim, start, end))

    print(f"Total combinations: {len(combos)}")
    for dim, start, end in combos:
        cmd = [
            sys.executable,
            str(EXTRACT),
            "--dimension",
            dim,
            "--start",
            str(start),
            "--end",
            str(end),
            "--num-layers",
            str(args.num_layers),
        ]
        if args.execute:
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        else:
            print(" ".join(cmd))


if __name__ == "__main__":
    main()
