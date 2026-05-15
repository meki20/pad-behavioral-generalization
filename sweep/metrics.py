"""OLS / RMSE helpers for multiplier vs PAD-axis analysis (mirrors Results UI linearTrend)."""
from __future__ import annotations

import numpy as np


def linear_trend(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """
    Ordinary least squares y ≈ intercept + slope * x on finite pairs.
    Returns (slope, intercept, rmse, mad).
    """
    pts: list[tuple[float, float]] = []
    for i in range(xs.size):
        x = float(xs[i])
        y = float(ys[i])
        if np.isfinite(x) and np.isfinite(y):
            pts.append((x, y))
    n = len(pts)
    if n < 2:
        y0 = float(pts[0][1]) if n == 1 else 0.0
        return 0.0, y0, 0.0, 0.0
    sx = sy = sxx = sxy = 0.0
    for x, y in pts:
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    sse = 0.0
    sabs = 0.0
    for x, y in pts:
        pred = intercept + slope * x
        e = y - pred
        sse += e * e
        sabs += abs(e)
    rmse = float(np.sqrt(sse / n))
    mad = float(sabs / n)
    return float(slope), float(intercept), rmse, mad


def norm_rmse(rmse: float) -> float:
    """Map RMSE to [0,1], higher is better (tighter fit)."""
    return float(1.0 / (1.0 + max(0.0, rmse)))


def norm_ols_slope(slope: float) -> float:
    """
    Expect steering to move the PAD axis in the positive direction as multiplier increases.
    Non-positive slopes map toward 0; strong positive slopes map toward 1.
    """
    return float(np.clip(max(0.0, slope) * 0.65, 0.0, 1.0))
