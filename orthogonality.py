#!/usr/bin/env python3
"""
Compare three steering-vector .pt files (steering_vectors.SteeringVector pickles).

Builds one flat vector per file by concatenating layer blocks in sorted layer-index
order; missing layers are zero-padded so different layer ranges still live in the
same ambient space. Prints pairwise and 3-way linear-algebra metrics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


def _load_steering(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _layer_activations(obj: object, path: Path) -> dict[int, torch.Tensor]:
    acts = getattr(obj, "layer_activations", None)
    if not isinstance(acts, dict) or not acts:
        raise TypeError(
            f"{type(obj).__name__} from {path}: expected attribute "
            "'layer_activations' to be a non-empty dict[int, Tensor] "
            "(steering_vectors.SteeringVector)."
        )
    return acts


def _validate_and_union_dims(
    paths: list[Path], objs: list[object]
) -> tuple[list[int], dict[int, int]]:
    """Return sorted union of layer indices and resolved flat size per layer."""
    acts_list = [_layer_activations(o, p) for o, p in zip(objs, paths, strict=True)]
    union_layers: set[int] = set()
    for acts in acts_list:
        union_layers.update(acts.keys())

    if not union_layers:
        raise ValueError("No layer indices found in any of the three files.")

    layer_dims: dict[int, int] = {}
    for L in sorted(union_layers):
        d: int | None = None
        for acts, p in zip(acts_list, paths, strict=True):
            if L not in acts:
                continue
            t = acts[L]
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{p}: layer {L} is not a Tensor ({type(t)})")
            n = int(t.numel())
            if d is None:
                d = n
            elif n != d:
                raise ValueError(
                    f"Layer {L}: inconsistent flat size {d} vs {n} "
                    f"(check {paths[0]} vs {p})."
                )
        if d is None:
            raise ValueError(f"Layer {L} appears in union but no file has it (bug).")
        layer_dims[L] = d

    return sorted(union_layers), layer_dims


def _flatten_union(
    acts: dict[int, torch.Tensor],
    layer_order: list[int],
    layer_dims: dict[int, int],
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for L in layer_order:
        d = layer_dims[L]
        if L in acts:
            t = acts[L].flatten().float().contiguous()
            if t.numel() != d:
                raise ValueError(f"Layer {L}: expected {d} elements, got {t.numel()}")
            parts.append(t)
        else:
            parts.append(torch.zeros(d, dtype=torch.float32))
    return torch.cat(parts, dim=0)


def _pair_metrics(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12
) -> dict[str, float]:
    na = float(torch.linalg.vector_norm(a).item())
    nb = float(torch.linalg.vector_norm(b).item())
    dot = float(torch.dot(a, b).item())
    denom = na * nb
    cos = dot / (denom + eps) if denom > eps else float("nan")
    cos_c = max(-1.0, min(1.0, cos))
    angle_rad = math.acos(cos_c)
    angle_deg = math.degrees(angle_rad)
    sin_a = math.sin(angle_rad)
    return {
        "norm_a": na,
        "norm_b": nb,
        "dot": dot,
        "cosine_similarity": cos,
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "sin_angle": sin_a,
        "abs_cosine": abs(cos_c),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Orthogonality / alignment metrics for three steering .pt files."
    )
    ap.add_argument("pt1", type=Path, help="First .pt (e.g. arousal_10_15.pt)")
    ap.add_argument("pt2", type=Path, help="Second .pt")
    ap.add_argument("pt3", type=Path, help="Third .pt")
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Numerical epsilon for cosine denominator.",
    )
    args = ap.parse_args()
    paths = [args.pt1, args.pt2, args.pt3]

    try:
        objs = [_load_steering(p) for p in paths]
    except Exception as e:
        print(f"Load error: {e}", file=sys.stderr)
        return 1

    acts_list = []
    for p, o in zip(paths, objs, strict=True):
        try:
            acts_list.append(_layer_activations(o, p))
        except TypeError as e:
            print(f"{e}", file=sys.stderr)
            return 1

    try:
        layer_order, layer_dims = _validate_and_union_dims(paths, objs)
    except Exception as e:
        print(f"Layout error: {e}", file=sys.stderr)
        return 1

    flats = [_flatten_union(acts, layer_order, layer_dims) for acts in acts_list]
    dim = int(flats[0].shape[0])
    for i, v in enumerate(flats):
        if v.shape[0] != dim:
            print(f"Internal error: vector {i} len {v.shape[0]} != {dim}", file=sys.stderr)
            return 1

    labels = [p.name for p in paths]
    print("=== Steering vector orthogonality ===\n")
    for i, p in enumerate(paths):
        keys = sorted(acts_list[i].keys())
        missing = [L for L in layer_order if L not in acts_list[i]]
        print(f"[{labels[i]}]  path: {p.resolve()}")
        print(f"    type: {type(objs[i]).__name__}")
        print(f"    layers present: {keys}")
        print(f"    zero-padded layers in union: {missing if missing else '(none)'}")
        print(f"    flat L2 norm: {float(torch.linalg.vector_norm(flats[i]).item()):.6g}")
        print()

    print("--- Union space (concat all layers in index order; missing = zeros) ---")
    print(f"    union layer indices: {layer_order}")
    print(f"    total flat dimension: {dim}\n")

    pairs = [(0, 1), (0, 2), (1, 2)]
    names = ["(1 vs 2)", "(1 vs 3)", "(2 vs 3)"]

    print("--- Pairwise metrics ---\n")
    for (i, j), tag in zip(pairs, names, strict=True):
        m = _pair_metrics(flats[i], flats[j], eps=args.eps)
        print(f"{tag}  {labels[i]}  vs  {labels[j]}")
        print(f"    dot(a,b)                 {m['dot']:.10g}")
        print(f"    ||a||, ||b||             {m['norm_a']:.10g}, {m['norm_b']:.10g}")
        print(f"    cosine similarity        {m['cosine_similarity']:.10g}")
        print(f"    |cosine| (0 = orthog.)  {m['abs_cosine']:.10g}")
        print(f"    angle (deg)              {m['angle_deg']:.6g}")
        print(f"    angle (rad)              {m['angle_rad']:.8g}")
        print(f"    sin(angle)               {m['sin_angle']:.10g}")
        print()

    # Gram matrix G_ij = <vi, vj>
    G = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            G[i, j] = torch.dot(flats[i].double(), flats[j].double())
    print("--- Gram matrix G_ij = <v_i, v_j> (rows/cols: files 1..3) ---")
    for row in range(3):
        print(
            "    "
            + "  ".join(f"{G[row, col].item():.10g}" for col in range(3)),
        )
    print()

    # Normalized Gram (cosines)
    C = torch.zeros(3, 3, dtype=torch.float64)
    norms = [torch.linalg.vector_norm(flats[k].double()).item() for k in range(3)]
    for i in range(3):
        for j in range(3):
            if norms[i] > args.eps and norms[j] > args.eps:
                C[i, j] = G[i, j] / (norms[i] * norms[j])
            else:
                C[i, j] = float("nan")
    print("--- Normalized Gram (cosine between flat vectors) ---")
    for row in range(3):
        print(
            "    "
            + "  ".join(
                f"{C[row, col].item():.10g}"
                if not math.isnan(float(C[row, col].item()))
                else "nan"
                for col in range(3)
            ),
        )
    print()

    # Triple: volume / parallelotope squared (Gram determinant for 3 vectors in R^d, d>=3)
    det = torch.linalg.det(G).item()
    print("--- 3-vector summary ---")
    print(f"    det(Gram)              {det:.10g}")
    print("    (|det| large => vectors span a fuller 3D parallelepiped in their span;")
    print("     near 0 => nearly coplanar / dependent in 3D subspace)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
