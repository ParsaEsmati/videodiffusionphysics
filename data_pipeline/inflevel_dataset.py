"""
InfLevel dataset: one ``.mp4`` per video, returned in the same sample format as
``IntPhysDataset`` so the same collate function applies.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers.utils import export_to_video
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_pipeline.data_modules import collate_intphys, print0


class InfLevelDataset(Dataset):
    """InfLevel videos, each decoded from one ``.mp4`` file.

    Args:
        data_dir:           Root of ``inflevel_lab`` (or ``inflevel_sim``).
        meta_path:          CSV from ``create_inflevel_csv.py`` with ``videoid``
                            (file path relative to ``data_dir``), ``label``
                            (1 = plausible) and an optional ``prompt`` column.
        data_frac:          Fraction of the rows to use.
        is_strict_loading:  Raise on a load failure instead of falling back to
                            a black placeholder.
        skip_missing_files: Draw another video when one fails to load.
        resolution_options: List of ``(width, height)``. The last entry sizes
                            the placeholder; the collate function does the
                            actual resize.
        max_frames:         ``None`` loads all frames, ``N`` the first N,
                            ``0`` none (``images`` is ``None``, prompts only).
    """

    def __init__(
        self,
        data_dir: str,
        meta_path: str,
        data_frac: float = 1.0,
        is_strict_loading: bool = False,
        skip_missing_files: bool = True,
        resolution_options: Optional[List[tuple]] = None,
        max_frames: Optional[int] = None,
    ):
        if not resolution_options:
            raise ValueError(
                "InfLevelDataset requires resolution_options=[(width, height)]."
            )

        super().__init__()

        self.data_dir = data_dir
        self.meta_path = meta_path
        self.data_frac = data_frac
        self.is_strict_loading = is_strict_loading
        self.skip_missing_files = skip_missing_files
        self.resolution_options = resolution_options
        self.default_res = self.resolution_options[-1]
        self.max_frames = max_frames
        self.missing_files: List[str] = []

        print0(
            f"[bold cyan][InfLevelDataset][/bold cyan] "
            f"data_dir={self.data_dir}  meta={self.meta_path}"
        )

        self._load_metadata()

    # ── metadata ─────────────────────────────────────────────────────────────

    def _load_metadata(self) -> None:
        metadata = pd.read_csv(
            self.meta_path,
            on_bad_lines="skip",
            encoding="utf-8",
            engine="python",
            sep=",",
        )

        if self.data_frac < 1.0:
            metadata = metadata.sample(frac=self.data_frac).reset_index(drop=True)

        metadata.dropna(subset=["videoid", "label"], inplace=True)

        if "prompt" not in metadata.columns:
            metadata["prompt"] = ""

        self.metadata = metadata

        print0(
            f"[bold cyan][InfLevelDataset][/bold cyan] "
            f"Loaded {len(self.metadata)} video(s)"
        )

    # ── video loading ────────────────────────────────────────────────────────

    def _video_path(self, sample: pd.Series) -> str:
        return os.path.join(self.data_dir, str(sample["videoid"]))

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Decode an ``.mp4`` with OpenCV into a ``(C, T, H, W)`` uint8 tensor (the first ``max_frames`` if set)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open {video_path}")

        frames: list[np.ndarray] = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if self.max_frames and len(frames) >= self.max_frames:
                break
        cap.release()

        if not frames:
            raise RuntimeError(f"cv2 read 0 frames from {video_path}")

        arr = np.stack(frames, axis=0)                     # (T, H, W, C) uint8
        return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item: int) -> Dict:
        width, height = self.default_res
        max_retries = 20 if self.skip_missing_files else 1

        for _ in range(max_retries):
            idx = item % len(self.metadata)
            sample = self.metadata.iloc[idx]
            video_path = self._video_path(sample)
            label = int(sample["label"])

            if self.max_frames == 0:
                prompt = str(sample.get("prompt", "") or "")
                return {"images": None, "paths": video_path,
                        "labels": label, "prompt": prompt}

            try:
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(f"video not found: {video_path}")
                imgs = self._load_video_frames(video_path)  # (C, T, H, W) uint8
            except Exception as exc:
                if video_path not in self.missing_files:
                    self.missing_files.append(video_path)
                if self.is_strict_loading:
                    raise
                print0(
                    f"[bold cyan][InfLevelDataset][/bold cyan] "
                    f"Load error for {video_path}: {exc}"
                )
                if not self.skip_missing_files:
                    imgs = torch.zeros(3, 1, height, width, dtype=torch.uint8)
                    break
                # Try another random sample
                item = int(np.random.randint(len(self)))
                continue

            prompt = str(sample.get("prompt", "") or "")
            return {"images": imgs, "paths": video_path,
                    "labels": label, "prompt": prompt}

        raise RuntimeError(
            f"InfLevelDataset: {max_retries} consecutive load failures. "
            f"Last attempt: {video_path}. Missing/corrupt entries so far: "
            f"{len(self.missing_files)}."
        )


# ============================================================================
# Standalone test: load one batch, save first sample as mp4
# ============================================================================


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            "Usage: python -m data_pipeline.inflevel_dataset "
            "<data_dir> <meta_csv> <width> <height> <num_frames> [output.mp4]"
        )
        sys.exit(1)

    data_dir   = sys.argv[1]
    meta_csv   = sys.argv[2]
    width      = int(sys.argv[3])
    height     = int(sys.argv[4])
    num_frames = int(sys.argv[5])
    out_path   = sys.argv[6] if len(sys.argv) > 6 else "test_batch.mp4"

    dataset = InfLevelDataset(
        data_dir=data_dir,
        meta_path=meta_csv,
        skip_missing_files=True,
        resolution_options=[(width, height)],
    )

    collate = partial(
        collate_intphys,
        resolution_options=[(width, height)],
        frame_count_options=[num_frames],
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
    batch  = next(iter(loader))
    images = batch["images"]
    labels = batch["labels"]
    paths  = batch["paths"]

    print(f"Batch images shape : {images.shape}")
    print(f"Batch labels       : {labels.tolist()}")
    print(f"Video paths        : {paths}")

    video_tensor = images[0]
    T = video_tensor.shape[1]
    pil_frames = [
        Image.fromarray(
            (video_tensor[:, t] * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy(),
            mode="RGB",
        )
        for t in range(T)
    ]
    export_to_video(pil_frames, out_path, fps=10)
    print(f"\nSaved {T}-frame video (label={labels[0].item()}) to {out_path}")
