"""Preset layer grids and multiplier lists for PAD sweep pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

Dimension = Literal["valence", "arousal", "dominance"]

# Default 32-layer sweep: inclusive end index last_layer.
FULL32_STARTS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
FULL32_WIDTHS = (1, 2, 4, 6, 8, 12)

MULTIPLIERS_VALENCE = (
    -0.6,
    -0.4,
    -0.25,
    -0.15,
    0.0,
    0.05,
    0.10,
    0.15,
    0.25,
)
MULTIPLIERS_AROUSAL = (-0.7, -0.5, -0.4, -0.25, 0.0, 0.15, 0.25, 0.4)
MULTIPLIERS_DOMINANCE = (-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25)

_PRESET_MULTS: dict[Dimension, tuple[float, ...]] = {
    "valence": MULTIPLIERS_VALENCE,
    "arousal": MULTIPLIERS_AROUSAL,
    "dominance": MULTIPLIERS_DOMINANCE,
}


@dataclass(frozen=True)
class SweepJob:
    """One steering vector identity plus its generation multiplier grid."""

    dimension: Dimension
    start: int
    end: int

    @property
    def range_id(self) -> str:
        return f"{self.dimension}_{self.start}_{self.end}"

    def multipliers(self) -> tuple[float, ...]:
        return _PRESET_MULTS[self.dimension]

    def multipliers_csv(self) -> str:
        return ",".join(str(m) for m in self.multipliers())


def iter_layer_ranges(
    *,
    starts: Iterable[int],
    widths: Iterable[int],
    last_layer: int,
) -> list[tuple[int, int]]:
    """Return unique (start, end) with end = start + width - 1 and end <= last_layer."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for s in starts:
        for w in widths:
            e = s + w - 1
            if e <= last_layer:
                key = (s, e)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def build_full32_jobs(
    dimensions: tuple[Dimension, ...] = ("valence", "arousal", "dominance"),
    *,
    num_layers: int = 32,
) -> list[SweepJob]:
    last = num_layers - 1
    ranges = iter_layer_ranges(starts=FULL32_STARTS, widths=FULL32_WIDTHS, last_layer=last)
    jobs: list[SweepJob] = []
    for dim in dimensions:
        for start, end in ranges:
            jobs.append(SweepJob(dimension=dim, start=start, end=end))
    return jobs
