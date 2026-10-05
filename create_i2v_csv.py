#!/usr/bin/env python3
"""
Build a CSV for image-to-video generation from a folder of first frames, with
a frame count per clip that matches the duration of the real clip.
"""

import csv
from pathlib import Path

# ============ CONFIGURATION ============
DATA_DIR = "/projects/b5bi/parsaest/videophys_results/data/real_world_first_frames"
PROMPT = "A toy car crashing into the blocks"
OUTPUT_CSV = "real_world_prompts_i2v.csv"

# CSV with real clip info. Expected columns: name, fps, num_frames
# name should match the image stem (e.g. "0OUuj3qJpkvM.mp4" matches "0OUuj3qJpkvM.jpg")
REAL_CLIPS_CSV = "real_clips_info.csv"

# WAN generation FPS (the --fps default of inference.py)
GEN_FPS = 16
# =======================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_clip_metadata(csv_path: Path) -> dict[str, dict]:
    """Load real clip metadata keyed by video stem (filename without extension)."""
    metadata = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["name"]).stem
            metadata[stem] = {
                "fps": float(row["fps"]),
                "num_frames": int(row["num_frames"]),
            }
    return metadata


def duration_matched_gen_frames(real_frames: int, real_fps: float, gen_fps: float) -> int:
    """Compute gen frames so that generated duration matches real clip duration."""
    return round(real_frames / real_fps * gen_fps)


def main():
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        print(f"Error: Data directory does not exist: {DATA_DIR}")
        return

    real_clips_path = Path(REAL_CLIPS_CSV)
    if not real_clips_path.exists():
        print(f"Error: Real clips metadata CSV not found: {REAL_CLIPS_CSV}")
        return

    metadata = load_clip_metadata(real_clips_path)
    print(f"Loaded metadata for {len(metadata)} clips")

    images = sorted([
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        print(f"No image files found in {DATA_DIR}")
        return

    print(f"Found {len(images)} images")

    rows = []
    missing = []
    for image_path in images:
        stem = image_path.stem
        clip_meta = metadata.get(stem)

        if clip_meta is None:
            missing.append(stem)
            target_frames = None
        else:
            target_frames = duration_matched_gen_frames(
                clip_meta["num_frames"], clip_meta["fps"], GEN_FPS
            )

        rows.append({
            "prompt": PROMPT,
            "output_name": f"gen_{stem}",
            "input_image": str(image_path),
            "num_frames": target_frames,
        })

    if missing:
        print(f"Warning: No metadata found for {len(missing)} images: {missing}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        for row in rows:
            f.write(f"{row['prompt']},{row['output_name']},{row['input_image']},{row['num_frames']}\n")

    print(f"Created {OUTPUT_CSV} with {len(rows)} entries")
    print(f"\n{'Image stem':<30} {'Real frames':>12} {'Real dur':>10} {'Gen frames':>12}")
    print("-" * 66)
    for row in rows:
        stem = Path(row["input_image"]).stem
        meta = metadata.get(stem)
        if meta:
            real_dur = meta["num_frames"] / meta["fps"]
            print(f"{stem:<30} {meta['num_frames']:>12} {real_dur:>9.2f}s {row['num_frames']:>11}")
        else:
            print(f"{stem:<30} {'N/A':>12} {'N/A':>10} {'N/A':>11}")


if __name__ == "__main__":
    main()
