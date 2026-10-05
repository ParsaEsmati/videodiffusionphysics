"""
IntPhys dataset (one folder of PNG frames per scene), the collate function
shared by all video datasets, and a LightningDataModule.
"""

from __future__ import annotations

import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from diffusers.utils import export_to_video
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms import v2


# ============================================================================
# Helpers
# ============================================================================


@rank_zero_only
def print0(*args, **kwargs):
    """Print on the rank-0 process only."""
    print(*args, **kwargs)


def resize_and_time_align(
    images: torch.Tensor,
    t_target: int,
    transform,
) -> torch.Tensor:
    """Bring a ``(C, T, H, W)`` video to ``t_target`` frames (uniform subsampling, or repeating the
    last frame when it is too short) and apply the spatial ``transform``."""
    C, T, H, W = images.shape

    # --- temporal alignment ---------------------------------------------------
    if T == t_target:
        pass  # nothing to do
    elif T > t_target:
        indices = torch.linspace(0, T - 1, t_target).long()
        images = images[:, indices]
    else:
        # pad by repeating the last frame
        pad_indices = torch.tensor(
            list(range(T)) + [T - 1] * (t_target - T)
        )
        images = images[:, pad_indices]

    # --- spatial transform ----------------------------------------------------
    # torchvision v2 transforms handle arbitrary leading dimensions,
    # so (C, T, H, W) is processed correctly without looping over frames.
    images = transform(images)

    return images  # (C, T_target, H_out, W_out)


def collate_intphys(
    samples: List[Dict],
    resolution_options: List[tuple],
    frame_count_options: List[int],
) -> Optional[Dict]:
    """Collate samples into ``images`` ``(B, C, T, H, W)`` float in [0, 1] (``None`` if no frames were
    loaded), ``paths``, ``labels`` and ``prompts``, at a resolution and frame count drawn from the options."""
    if not samples:
        return None

    w_target, h_target = random.choice(resolution_options)
    t_target = random.choice(frame_count_options)

    transform = v2.Compose([
        v2.Resize(h_target, antialias=True),
        v2.CenterCrop((h_target, w_target)),
    ])

    # images is None when load_images=False (e.g. T2V generation needs only prompts)
    if samples[0]["images"] is not None:
        images = torch.stack(
            [resize_and_time_align(s["images"], t_target, transform) for s in samples],
            dim=0,
        )  # (B, C, T, H, W) uint8
        images = images.float() / 255.0  # → float32 [0, 1], cast once per batch
    else:
        images = None

    paths = [s["paths"] for s in samples]
    labels = torch.tensor([s["labels"] for s in samples], dtype=torch.long)
    prompts = [s["prompt"] for s in samples]

    return {"images": images, "paths": paths, "labels": labels, "prompts": prompts}


def worker_init_fn(_) -> None:
    """Seed each DataLoader worker differently for reproducible randomness."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, IterableDataset):
        split_size = dataset.num_records // worker_info.num_workers
        dataset.sample_ids = dataset.valid_ids[
            worker_id * split_size : (worker_id + 1) * split_size
        ]

    current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    np.random.seed(np.random.get_state()[1][current_id] + worker_id)


# ============================================================================
# Dataset
# ============================================================================


class IntPhysDataset(Dataset):
    """IntPhys scenes, each loaded from the PNG frames in ``<scene_dir>/scene/``.

    Args:
        data_dir:           Root directory of the dataset.
        meta_path:          CSV with ``videoid`` (scene folder relative to
                            ``data_dir``), ``label`` (1 = plausible) and an
                            optional ``prompt`` column.
        data_frac:          Fraction of the rows to use.
        is_strict_loading:  Raise on a load failure instead of falling back to
                            a black placeholder.
        skip_missing_files: Draw another scene when one is missing.
        resolution_options: List of ``(width, height)``. The last entry sizes
                            the placeholder; the collate function does the
                            actual resize.
        max_frames:         ``None`` loads all frames, ``1`` the first frame,
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
                "IntPhysDataset requires resolution_options to be set explicitly, "
                "e.g. resolution_options=[(width, height)]. "
                "This ensures the data resolution matches the pipeline."
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
            f"[bold yellow][IntPhysDataset][/bold yellow] "
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

        # Ensure prompt column exists (fill with empty string when absent)
        if "prompt" not in metadata.columns:
            metadata["prompt"] = ""

        self.metadata = metadata

        print0(
            f"[bold yellow][IntPhysDataset][/bold yellow] "
            f"Loaded {len(self.metadata)} scene(s)"
        )

    # ── path helpers ─────────────────────────────────────────────────────────

    def _get_scene_dir(self, sample: pd.Series) -> str:
        """Return the absolute path to the IntPhys scene leaf directory."""
        rel_path = str(sample["videoid"])
        return os.path.join(self.data_dir, rel_path)

    def _load_scene_frames(self, scene_dir: str) -> torch.Tensor:
        """Load the PNG frames of a scene as a ``(C, T, H, W)`` uint8 tensor (the first ``max_frames`` if set)."""
        png_dir = os.path.join(scene_dir, "scene")
        png_files = sorted(Path(png_dir).glob("*.png"))

        if self.max_frames is not None:
            png_files = png_files[: self.max_frames]

        frames: List[torch.Tensor] = []
        for p in png_files:
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)  # (H, W, 3) uint8 – 4x lighter than float32
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (C, H, W) uint8

        # (T, C, H, W) → (C, T, H, W)
        return torch.stack(frames, dim=0).permute(1, 0, 2, 3)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item: int) -> Dict:
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]

        scene_dir = self._get_scene_dir(sample)
        label = int(sample["label"])
        width, height = self.default_res

        imgs = None
        if self.max_frames != 0:  # 0 means "no frames needed" (e.g. T2V generation)
            try:
                png_dir = os.path.join(scene_dir, "scene")
                if not os.path.isdir(png_dir):
                    raise FileNotFoundError(f"scene/ directory not found: {png_dir}")

                imgs = self._load_scene_frames(scene_dir)  # (C, T, H, W) uint8

            except Exception as exc:
                if scene_dir not in self.missing_files:
                    self.missing_files.append(scene_dir)

                if not isinstance(exc, FileNotFoundError):
                    if self.is_strict_loading:
                        raise
                    print0(
                        f"[bold yellow][IntPhysDataset][/bold yellow] "
                        f"Load error for {scene_dir}: {exc}"
                    )

                if self.skip_missing_files:
                    print0(
                        f"[bold yellow][IntPhysDataset][/bold yellow] "
                        f"Missing/corrupt scene {scene_dir}. Re-sampling."
                    )
                    return self.__getitem__(np.random.randint(len(self)))

                # Fall back to a black video (1 frame placeholder, uint8)
                print0(
                    "[bold yellow][IntPhysDataset][/bold yellow] "
                    "Using black-frame placeholder."
                )
                imgs = torch.zeros(3, 1, height, width, dtype=torch.uint8)

        prompt = str(sample.get("prompt", "") or "")
        return {"images": imgs, "paths": scene_dir, "labels": label, "prompt": prompt}


# ============================================================================
# LightningDataModule
# ============================================================================


class IntPhysDataModule(pl.LightningDataModule):
    """LightningDataModule around ``IntPhysDataset``.

    Args:
        config: Dict with ``batch_size``, ``num_workers``, ``pin_memory``,
                ``drop_last``, ``persistent_workers`` and a ``dataset_params``
                dict holding ``data_dir``, ``train_meta_path``, an optional
                ``val_meta_path``, ``resolution_options``,
                ``frame_count_options`` and the ``IntPhysDataset`` options.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config or {}

        self.batch_size = self.config.get("batch_size", 1)
        self.num_workers = self.config.get("num_workers", 0)
        self.pin_memory = self.config.get("pin_memory", True)
        self.drop_last = self.config.get("drop_last", False)
        self.persistent_workers = self.config.get(
            "persistent_workers", self.num_workers > 0
        )

        ds_params = self.config.get("dataset_params") or {}
        self.data_dir = ds_params["data_dir"]
        self.train_meta_path = ds_params["train_meta_path"]

        self.data_frac = ds_params.get("data_frac", 1.0)
        self.is_strict_loading = ds_params.get("is_strict_loading", False)
        self.skip_missing_files = ds_params.get("skip_missing_files", True)
        if "resolution_options" not in ds_params:
            raise ValueError(
                "IntPhysDataModule config must include dataset_params.resolution_options, "
                "e.g. resolution_options: [[832, 480]]"
            )
        if "frame_count_options" not in ds_params:
            raise ValueError(
                "IntPhysDataModule config must include dataset_params.frame_count_options, "
                "e.g. frame_count_options: [81]"
            )
        self.resolution_options = ds_params["resolution_options"]
        self.frame_count_options = ds_params["frame_count_options"]

        self.train_set: Optional[IntPhysDataset] = None
        self.val_set: Optional[IntPhysDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "train"):
            self.train_set = IntPhysDataset(
                data_dir=self.data_dir,
                meta_path=self.train_meta_path,
                data_frac=self.data_frac,
                is_strict_loading=self.is_strict_loading,
                skip_missing_files=self.skip_missing_files,
                resolution_options=self.resolution_options,
            )

        ds_params = self.config.get("dataset_params") or {}
        val_meta_path = ds_params.get("val_meta_path")
        if val_meta_path:
            self.val_set = IntPhysDataset(
                data_dir=self.data_dir,
                meta_path=val_meta_path,
                data_frac=ds_params.get("val_data_frac", 1.0),
                is_strict_loading=self.is_strict_loading,
                skip_missing_files=self.skip_missing_files,
                resolution_options=self.resolution_options,
            )

    def _make_collate(self) -> callable:
        return partial(
            collate_intphys,
            resolution_options=self.resolution_options,
            frame_count_options=self.frame_count_options,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None, (
            "Call setup('fit') or setup(None) before requesting train_dataloader."
        )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=self._make_collate(),
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_set is None:
            return None
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=self._make_collate(),
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    """Load one batch and save its first sample as an mp4."""
    if len(sys.argv) < 6:
        print(
            "Usage: python -m data_pipeline.data_modules "
            "<data_dir> <meta_csv> <width> <height> <num_frames> [output.mp4]"
        )
        sys.exit(1)

    data_dir   = sys.argv[1]
    meta_csv   = sys.argv[2]
    width      = int(sys.argv[3])
    height     = int(sys.argv[4])
    num_frames = int(sys.argv[5])
    out_path   = sys.argv[6] if len(sys.argv) > 6 else "test_batch.mp4"

    # ── build dataset & loader (no Lightning, just plain DataLoader) ──────────
    dataset = IntPhysDataset(
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

    # ── fetch one batch ───────────────────────────────────────────────────────
    batch = next(iter(loader))
    images = batch["images"]   # (B, C, T, H, W)  float in [0, 1]
    labels = batch["labels"]   # (B,)
    paths  = batch["paths"]

    print(f"Batch images shape : {images.shape}")
    print(f"Batch labels       : {labels.tolist()}")
    print(f"Scene paths        : {paths}")

    # ── save first sample as mp4 ──────────────────────────────────────────────
    # images[0]: (C, T, H, W) → per-frame (H, W, C) uint8 PIL Images
    video_tensor = images[0]              # (C, T, H, W)
    T = video_tensor.shape[1]

    pil_frames = []
    for t in range(T):
        frame = video_tensor[:, t, :, :]  # (C, H, W)
        frame_uint8 = (frame * 255).clamp(0, 255).byte()
        frame_np = frame_uint8.permute(1, 2, 0).numpy()  # (H, W, C)
        pil_frames.append(Image.fromarray(frame_np, mode="RGB"))

    export_to_video(pil_frames, out_path, fps=10)
    print(f"\nSaved {T}-frame video (label={labels[0].item()}) to {out_path}")
