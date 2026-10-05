#!/usr/bin/env python3
"""
Build the InfLevel CSVs: one with every video and one per principle, with the
plausibility label derived from the event code in each file name.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional


PRINCIPLES = ("continuity", "gravity", "solidity")


def _parse_filename(name: str) -> Optional[dict]:
    """Parse an InfLevel file name into its parts: six for continuity (the last two are the event
    code and the direction), five for gravity and solidity. Returns ``None`` if it does not fit."""
    stem = Path(name).stem
    parts = stem.split("__")

    if len(parts) == 6:
        view, principle, obj_a, obj_b, code, direction = parts
        if len(code) != 2 or set(code) - {"v", "i"}:
            return None
        return {
            "view":        view,
            "principle":   principle,
            "object_a":    obj_a,
            "object_b":    obj_b,
            "event_code":  code,
            "cover_state": "",
            "direction":   direction,
        }

    if len(parts) == 5:
        view, principle, obj_a, obj_b, code = parts
        if len(code) != 2 or code[0] not in ("c", "u") or code[1] not in ("v", "i"):
            return None
        return {
            "view":        view,
            "principle":   principle,
            "object_a":    obj_a,
            "object_b":    obj_b,
            "event_code":  code,
            "cover_state": "covered" if code[0] == "c" else "uncovered",
            "direction":   "",
        }

    return None


def _label_for(principle: str, code: str) -> int:
    """Label from the event code: continuity ``vv``/``ii`` and gravity/solidity ``cv``/``ui`` are
    plausible (1); the other codes are impossible (0)."""
    if principle == "continuity":
        return 1 if code in ("vv", "ii") else 0
    return 1 if code in ("cv", "ui") else 0


def _iter_videos(data_dir: Path) -> Iterator[Path]:
    """Yield .mp4 files under continuity/gravity/solidity, skipping macOS junk."""
    for principle in PRINCIPLES:
        pdir = data_dir / principle
        if not pdir.is_dir():
            continue
        for p in sorted(pdir.iterdir()):
            if p.suffix != ".mp4" or p.name.startswith("._"):
                continue
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an InfLevel CSV of videos with derived labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_dir", type=Path,
        help="Root inflevel_{lab,sim} dir (contains continuity/ gravity/ solidity/)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("inflevel.csv"),
        help="Output CSV path (default: inflevel.csv)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    FIELDS = [
        "videoid", "label", "principle", "view",
        "object_a", "object_b", "event_code", "cover_state", "direction",
    ]

    rows: list[dict] = []
    code_counts: Counter = Counter()
    skipped = 0

    for p in _iter_videos(data_dir):
        meta = _parse_filename(p.name)
        if meta is None:
            skipped += 1
            print(f"  WARNING  unparseable filename: {p.name}", file=sys.stderr)
            continue

        principle = meta["principle"]
        code = meta["event_code"]
        code_counts[(principle, code)] += 1

        rows.append({
            "videoid": str(p.relative_to(data_dir)),
            "label":   _label_for(principle, code),
            **meta,
        })

    if not rows:
        print("Error: no valid videos found.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _write_csv(path: Path, rs: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rs)

    # Combined CSV (all principles)
    _write_csv(args.output, rows)

    # One CSV per principle, named <stem>_<principle>.<suffix>
    stem, suffix = args.output.stem, args.output.suffix
    parent = args.output.parent
    for principle in sorted({r["principle"] for r in rows}):
        sub = [r for r in rows if r["principle"] == principle]
        sub_path = parent / f"{stem}_{principle}{suffix}"
        _write_csv(sub_path, sub)
        n1 = sum(r["label"] == 1 for r in sub)
        n0 = sum(r["label"] == 0 for r in sub)
        print(f"Wrote {len(sub)} rows to {sub_path}  (plausible={n1}, impossible={n0})")

    # ── summary ──────────────────────────────────────────────────────────────
    n_pos = sum(r["label"] == 1 for r in rows)
    n_neg = sum(r["label"] == 0 for r in rows)
    principles_seen = sorted({r["principle"] for r in rows})

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  Plausible   : {n_pos}")
    print(f"  Impossible  : {n_neg}")
    print(f"  Principles  : {', '.join(principles_seen)}")
    print(f"  Event codes (principle, code → count):")
    for (pr, code), n in sorted(code_counts.items()):
        print(f"      {pr:11s} {code:2s}  {n}")
    if skipped:
        print(f"  Skipped (unparseable): {skipped}")


if __name__ == "__main__":
    main()
