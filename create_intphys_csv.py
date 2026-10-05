#!/usr/bin/env python3
"""
Build the IntPhys CSV: one row per scene, with the plausibility label read
from the scene's ``status.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterator


# ============================================================================
# Scene discovery
# ============================================================================


def _iter_scenes(data_dir: Path) -> Iterator[Path]:
    """Yield the scene folders (``O1/01/1`` ...) in sorted order, so the CSV order is stable."""
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith("O"):
            continue

        for scenario_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not scenario_dir.is_dir():
                continue

            for condition_dir in sorted(scenario_dir.iterdir(), key=lambda p: p.name):
                if not condition_dir.is_dir():
                    continue

                scene_dir = condition_dir / "scene"
                status_file = condition_dir / "status.json"

                if scene_dir.is_dir() and status_file.is_file():
                    yield condition_dir


# ============================================================================
# Row builder
# ============================================================================


def _build_row(leaf: Path, data_dir: Path) -> dict:
    """Read metadata for one scene and return a CSV row dict."""
    with open(leaf / "status.json") as f:
        status = json.load(f)

    is_possible: bool = status["header"]["is_possible"]
    label = 1 if is_possible else 0

    png_count = len(list((leaf / "scene").glob("*.png")))

    # Derive structural fields from path components
    # Expected relative path:  O1/01/1
    rel = leaf.relative_to(data_dir)
    parts = rel.parts  # ('O1', '01', '1')

    return {
        "videoid":    str(rel),
        "label":      label,
        "split":      parts[0] if len(parts) > 0 else "",
        "scenario":   parts[1] if len(parts) > 1 else "",
        "condition":  parts[2] if len(parts) > 2 else "",
        "num_frames": png_count,
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an IntPhys CSV from a dev directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root dev directory (contains O1/, O2/, O3/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("intphys_dev.csv"),
        help="Output CSV path (default: intphys_dev.csv)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    FIELDS = ["videoid", "label", "split", "scenario", "condition", "num_frames"]

    rows = []
    errors = []

    for leaf in _iter_scenes(data_dir):
        try:
            rows.append(_build_row(leaf, data_dir))
        except Exception as exc:
            errors.append((leaf, exc))
            print(f"  WARNING  {leaf}: {exc}", file=sys.stderr)

    if not rows:
        print("Error: no valid scenes found.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ── summary ───────────────────────────────────────────────────────────────
    n_possible = sum(1 for r in rows if r["label"] == 1)
    n_impossible = sum(1 for r in rows if r["label"] == 0)

    splits = sorted({r["split"] for r in rows})
    frame_counts = sorted({r["num_frames"] for r in rows})

    print(f"Wrote {len(rows)} scenes to {args.output}")
    print(f"  Plausible     : {n_possible}")
    print(f"  Not plausible : {n_impossible}")
    print(f"  Splits        : {', '.join(splits)}")
    print(f"  Frame counts  : {frame_counts}")
    if errors:
        print(f"  Skipped (errors): {len(errors)}")


if __name__ == "__main__":
    main()
