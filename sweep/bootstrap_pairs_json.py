"""One-off: read pair lists from legacy extract_*.py via AST, write sweep/pairs/*.json."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = ROOT / "sweep" / "pairs"


def _extract_assign_list(py_path: Path, var_name: str) -> list:
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == var_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"{var_name} not found in {py_path}")


def main() -> None:
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("valence_pairs", ROOT / "extract_valence_pairs.py", "valence.json"),
        ("arousal_pairs", ROOT / "extract_arousal_pairs.py", "arousal.json"),
        ("dominance_pairs", ROOT / "extract_dominance_pairs.py", "dominance.json"),
    ]
    for var, src, out_name in mapping:
        data = _extract_assign_list(src, var)
        out = PAIRS_DIR / out_name
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out} ({len(data)} pairs)")


if __name__ == "__main__":
    main()
