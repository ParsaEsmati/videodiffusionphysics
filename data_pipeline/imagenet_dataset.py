"""
ImageNet images loaded as one-frame videos ``(C, 1, H, W)``, in the same sample
format as ``IntPhysDataset``. The label is the class index.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_pipeline.data_modules import (
    collate_intphys,
    print0,
    worker_init_fn,
)


# ============================================================================
# Dataset
# ============================================================================


class ImageNetDataset(Dataset):
    """ImageNet images, each returned as a one-frame video.

    Args:
        data_dir:           Root directory; relative ``image_path`` values in
                            the CSV are resolved against it.
        meta_path:          CSV with ``image_path`` and ``label`` (class index);
                            ``prompt`` and ``synset`` are optional.
        data_frac:          Fraction of the rows to use.
        is_strict_loading:  Raise on a load failure instead of falling back to
                            a black placeholder.
        skip_missing_files: Draw another image when one is missing.
        resolution_options: List of ``(width, height)``. The last entry sizes
                            the placeholder.
        max_frames:         Ignored (always one frame); kept so the signature
                            matches ``IntPhysDataset``.
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
                "ImageNetDataset requires resolution_options to be set, "
                "e.g. resolution_options=[(width, height)]."
            )

        super().__init__()
        self.data_dir = data_dir
        self.meta_path = meta_path
        self.data_frac = data_frac
        self.is_strict_loading = is_strict_loading
        self.skip_missing_files = skip_missing_files
        self.resolution_options = resolution_options
        self.default_res = self.resolution_options[-1]
        # max_frames is part of the interface but always treated as 1 here;
        # ignoring 0 / None / >1 is intentional — an ImageNet "video" has
        # exactly one frame.
        self.max_frames = max_frames
        self.missing_files: List[str] = []

        print0(
            f"[bold cyan][ImageNetDataset][/bold cyan] "
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

        required = {"image_path", "label"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(
                f"ImageNetDataset CSV missing required columns: {sorted(missing)}. "
                f"Saw: {list(metadata.columns)}"
            )

        if self.data_frac < 1.0:
            metadata = metadata.sample(frac=self.data_frac).reset_index(drop=True)

        metadata.dropna(subset=["image_path", "label"], inplace=True)
        # Ensure optional columns exist (filled with sane defaults) so
        # downstream code can rely on their presence.
        if "prompt" not in metadata.columns:
            metadata["prompt"] = ""
        if "synset" not in metadata.columns:
            metadata["synset"] = ""

        self.metadata = metadata
        print0(
            f"[bold cyan][ImageNetDataset][/bold cyan] "
            f"Loaded {len(self.metadata)} image(s) across "
            f"{self.metadata['label'].nunique()} distinct labels"
        )

    # ── path helpers ─────────────────────────────────────────────────────────

    def _resolve_image_path(self, sample: pd.Series) -> str:
        """Resolve ``image_path``: absolute paths win, otherwise join under
        ``data_dir``."""
        p = str(sample["image_path"])
        if os.path.isabs(p):
            return p
        return os.path.join(self.data_dir, p)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item: int) -> Dict:
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]

        img_path = self._resolve_image_path(sample)
        label = int(sample["label"])
        width, height = self.default_res

        imgs: Optional[torch.Tensor] = None
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img, dtype=np.uint8)               # (H, W, 3)
            # (H, W, 3) → (3, H, W) → (3, 1, H, W) — single-frame "video".
            imgs = (
                torch.from_numpy(arr)
                .permute(2, 0, 1)
                .unsqueeze(1)
                .contiguous()
            )
        except Exception as exc:
            if img_path not in self.missing_files:
                self.missing_files.append(img_path)

            if self.is_strict_loading:
                raise

            if self.skip_missing_files:
                print0(
                    f"[bold cyan][ImageNetDataset][/bold cyan] "
                    f"Missing/corrupt image {img_path} ({exc}). Re-sampling."
                )
                return self.__getitem__(np.random.randint(len(self)))

            print0(
                f"[bold cyan][ImageNetDataset][/bold cyan] "
                f"Using black-frame placeholder for {img_path}: {exc}"
            )
            imgs = torch.zeros(3, 1, height, width, dtype=torch.uint8)

        prompt = str(sample.get("prompt", "") or "")

        # ``paths`` is fed to inference.py's scene-stem builder, which takes
        # the last 3 path parts. We construct a path that yields a clean
        # scene name like ``train_n01440764_n01440764_18`` for the typical
        # ImageNet layout while preserving traceability back to the source.
        scene_path = str(Path(img_path).with_suffix(""))

        return {
            "images": imgs,
            "paths": scene_path,
            "labels": label,
            "prompt": prompt,
        }


# ============================================================================
# LightningDataModule
# ============================================================================


class ImageNetDataModule(pl.LightningDataModule):
    """LightningDataModule around ``ImageNetDataset``.

    Args:
        config: Dict in the same form as for ``IntPhysDataModule``. Set
                ``dataset_params.frame_count_options`` to ``[1]``.
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

        ds = self.config.get("dataset_params") or {}
        self.data_dir = ds["data_dir"]
        self.train_meta_path = ds["train_meta_path"]
        self.data_frac = ds.get("data_frac", 1.0)
        self.is_strict_loading = ds.get("is_strict_loading", False)
        self.skip_missing_files = ds.get("skip_missing_files", True)

        if "resolution_options" not in ds:
            raise ValueError(
                "ImageNetDataModule needs dataset_params.resolution_options, "
                "e.g. resolution_options: [[832, 480]]."
            )
        if "frame_count_options" not in ds:
            raise ValueError(
                "ImageNetDataModule needs dataset_params.frame_count_options. "
                "For images use [1]; the causal VAE accepts T=1."
            )
        self.resolution_options = ds["resolution_options"]
        self.frame_count_options = ds["frame_count_options"]
        if self.frame_count_options != [1]:
            print0(
                f"[bold yellow][ImageNetDataModule][/bold yellow] "
                f"frame_count_options={self.frame_count_options}; the dataset "
                f"yields T=1, so the collator will replicate the single image "
                f"across the target frame count. Usually you want [1] here."
            )

        self.train_set: Optional[ImageNetDataset] = None
        self.val_set: Optional[ImageNetDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "train"):
            self.train_set = ImageNetDataset(
                data_dir=self.data_dir,
                meta_path=self.train_meta_path,
                data_frac=self.data_frac,
                is_strict_loading=self.is_strict_loading,
                skip_missing_files=self.skip_missing_files,
                resolution_options=self.resolution_options,
            )
        ds = self.config.get("dataset_params") or {}
        val_meta_path = ds.get("val_meta_path")
        if val_meta_path:
            self.val_set = ImageNetDataset(
                data_dir=self.data_dir,
                meta_path=val_meta_path,
                data_frac=ds.get("val_data_frac", 1.0),
                is_strict_loading=self.is_strict_loading,
                skip_missing_files=self.skip_missing_files,
                resolution_options=self.resolution_options,
            )

    def _make_collate(self):
        return partial(
            collate_intphys,
            resolution_options=self.resolution_options,
            frame_count_options=self.frame_count_options,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None
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
# CSV builder helper — scan a directory tree → emit metadata CSV
# ============================================================================


def build_imagenet_csv(
    data_dir: str | Path,
    out_csv: str | Path,
    splits: Optional[List[str]] = None,
    synset_to_idx: Optional[Dict[str, int]] = None,
    synset_to_name: Optional[Dict[str, str]] = None,
    image_glob: str = "*.JPEG",
    n_per_class: Optional[int] = None,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Scan ``data_dir/<split>/<synset>/`` for images and write a CSV with ``image_path``, ``label``,
    ``synset``, ``prompt`` and ``split``; ``n_per_class`` limits the images taken per class."""
    data_dir = Path(data_dir)
    splits = splits or ["train"]
    rows: List[Dict] = []
    rng = np.random.default_rng(rng_seed)

    # Build deterministic synset → index map if not provided.
    if synset_to_idx is None:
        synsets: set = set()
        for split in splits:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            for synset_dir in split_dir.iterdir():
                if synset_dir.is_dir():
                    synsets.add(synset_dir.name)
        synset_to_idx = {s: i for i, s in enumerate(sorted(synsets))}

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  WARN: split dir not found: {split_dir}")
            continue
        for synset_dir in sorted(split_dir.iterdir()):
            if not synset_dir.is_dir():
                continue
            synset = synset_dir.name
            if synset not in synset_to_idx:
                # Not in the user-supplied map — skip.
                continue
            label = synset_to_idx[synset]
            images = sorted(synset_dir.glob(image_glob))
            if n_per_class is not None and len(images) > n_per_class:
                pick = rng.choice(len(images), size=n_per_class, replace=False)
                images = [images[i] for i in sorted(pick)]
            prompt = (synset_to_name or {}).get(synset, synset)
            for p in images:
                rows.append({
                    "image_path": str(p.relative_to(data_dir)),
                    "label":      int(label),
                    "synset":     synset,
                    "prompt":     prompt,
                    "split":      split,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(
        f"Wrote {len(df)} ImageNet rows ({df['label'].nunique()} classes) "
        f"to {out_csv}"
    )
    return df


# ============================================================================
# Standalone smoke test
# ============================================================================


if __name__ == "__main__":
    """Quick check: load two images and print the batch shapes."""
    if len(sys.argv) < 5:
        print("Usage: python -m data_pipeline.imagenet_dataset "
              "<data_dir> <meta_csv> <width> <height>")
        sys.exit(1)
    data_dir, meta_csv, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

    ds = ImageNetDataset(
        data_dir=data_dir, meta_path=meta_csv,
        resolution_options=[(w, h)],
    )
    collate = partial(collate_intphys,
                      resolution_options=[(w, h)],
                      frame_count_options=[1])
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    print(f"images shape : {batch['images'].shape}   "
          f"(expect (B, 3, 1, {h}, {w}))")
    print(f"labels       : {batch['labels'].tolist()}")
    print(f"paths        : {batch['paths']}")
