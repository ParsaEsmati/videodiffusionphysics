"""
Datasets that load the saved block features for the probes, and the helpers
that index a run folder into a CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Debug: replace loaded features with deterministic random noise of the same shape.
# Trivial-classifiability sanity check — remove before release.
_DEBUG_RANDOM_FEATURES = False


class ProbeDataset(Dataset):
    """Saved block features of one scene per item, with its plausibility label.

    Each item is ``block_outputs.pt`` averaged over tokens: ``(num_blocks, D)``.

    Args:
        csv_path:      CSV with one row per scene.
        scene_dir_col: Column with the folder that holds ``block_outputs.pt``.
        label_col:     Column with the label (1 = plausible).
    """

    def __init__(
        self,
        csv_path: str | Path,
        scene_dir_col: str = "scene_dir",
        label_col: str = "label",
    ):
        df = pd.read_csv(csv_path)
        self.scene_dirs = df[scene_dir_col].tolist()
        self.labels = df[label_col].tolist()

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        scene_dir = Path(self.scene_dirs[idx])
        feat = torch.load(scene_dir / "block_outputs.pt", weights_only=True)  # (num_blocks, seq, D)
        # if feat.dim() == 3 and feat.shape[1] != 1000:
        #     seq_idx = torch.linspace(0, feat.shape[1] - 1, 1000).long()
        #     feat = feat[:, seq_idx, :]  # (num_blocks, 1000, D)
        if _DEBUG_RANDOM_FEATURES:
            g = torch.Generator().manual_seed(idx)
            feat = torch.randn(feat.shape, generator=g, dtype=feat.dtype)
        label = int(self.labels[idx])
        return feat.mean(dim=1), label


# ============================================================================
# Utility: build a probe CSV by scanning an inference output directory
# ============================================================================


def build_probe_csv(
    output_dir: str | Path,
    out_csv: str | Path,
    step: int = 0,
    legacy: bool = False,
) -> pd.DataFrame:
    """Index a run folder into a CSV with ``scene_dir`` and ``label``, keeping the scenes that have both
    ``meta.json`` and ``step_NNNN/block_outputs.pt`` (``legacy``: features directly in the scene folder)."""
    output_dir = Path(output_dir)
    step_name = f"step_{step:04d}"
    rows = []

    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        meta_path = scene_dir / "meta.json"
        if legacy:
            feat_dir = scene_dir
        else:
            feat_dir = scene_dir / step_name
        pt_path = feat_dir / "block_outputs.pt"
        if not (meta_path.exists() and pt_path.exists()):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        rows.append({"scene_dir": str(feat_dir), "label": int(meta["label"])})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    tag = "legacy" if legacy else f"step {step}"
    print(f"Wrote {len(df)} scenes ({tag}) to {out_csv}")
    return df


# ============================================================================
# Regression flavour — sibling of ProbeDataset / build_probe_csv for predicting
# continuous targets (e.g. PhyWorld initial conditions [x0, v0]).
# ============================================================================


def build_regress_csv(
    output_dir: str | Path,
    out_csv: str | Path,
    step: int = 0,
    legacy: bool = False,
    target_keys: tuple[str, ...] = ("x0", "v0"),
) -> pd.DataFrame:
    """Like ``build_probe_csv``, but with the regression targets (``x0``, ``v0``) read from
    ``init_conditions`` in each scene's ``meta.json``."""
    output_dir = Path(output_dir)
    step_name = f"step_{step:04d}"
    rows = []

    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        meta_path = scene_dir / "meta.json"
        feat_dir = scene_dir if legacy else scene_dir / step_name
        pt_path = feat_dir / "block_outputs.pt"
        if not (meta_path.exists() and pt_path.exists()):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        ic = meta.get("init_conditions")
        if ic is None or len(ic) < len(target_keys):
            continue
        row = {"scene_dir": str(feat_dir)}
        for i, k in enumerate(target_keys):
            row[k] = float(ic[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    tag = "legacy" if legacy else f"step {step}"
    print(f"Wrote {len(df)} regression scenes ({tag}) to {out_csv}")
    return df


class RegressDataset(Dataset):
    """Saved block features of one scene per item, with continuous targets.

    Args:
        csv_path:      CSV with one row per scene.
        scene_dir_col: Column with the folder that holds ``block_outputs.pt``.
        target_cols:   Columns with the regression targets.
    """

    def __init__(
        self,
        csv_path: str | Path,
        scene_dir_col: str = "scene_dir",
        target_cols: tuple[str, ...] = ("x0", "v0"),
    ):
        df = pd.read_csv(csv_path)
        self.scene_dirs = df[scene_dir_col].tolist()
        self.target_cols = tuple(target_cols)
        self.targets = df[list(self.target_cols)].values.astype("float32")  # (N, K)

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene_dir = Path(self.scene_dirs[idx])
        feat = torch.load(scene_dir / "block_outputs.pt", weights_only=True)  # (num_blocks, seq, D)
        if _DEBUG_RANDOM_FEATURES:
            g = torch.Generator().manual_seed(idx)
            feat = torch.randn(feat.shape, generator=g, dtype=feat.dtype)
        target = torch.from_numpy(self.targets[idx])
        return feat.mean(dim=1), target


# ============================================================================
# CLI timing test
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time .pt feature loading")
    parser.add_argument("csv", type=Path, help="Probe CSV (scene_dir, label)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    dataset = ProbeDataset(args.csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Dataset: {len(dataset)} scenes  |  batch_size={args.batch_size}  |  workers={args.num_workers}")

    # Warm-up
    _ = next(iter(loader))

    t0 = time.perf_counter()
    n_samples = 0
    for x, y in loader:
        n_samples += x.shape[0]
    elapsed = time.perf_counter() - t0

    print(f"Loaded {n_samples} samples in {elapsed:.3f}s  ({n_samples / elapsed:.1f} samples/s)")
    print(f"Feature shape: {x.shape}  dtype: {x.dtype}")
