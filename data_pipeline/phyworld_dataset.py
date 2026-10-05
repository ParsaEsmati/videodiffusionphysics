"""
PhyWorld parabola dataset: videos, initial conditions ``(x0, v0)`` and
trajectories stored in one HDF5 file (``magicr/phyworld`` on Hugging Face),
returned in the same sample format as ``IntPhysDataset``.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.data_modules import collate_intphys

# Optional packages: PyAV or OpenCV decodes the videos, imageio saves previews.
try:
    import av
except ImportError:
    av = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import imageio.v3 as iio
except ImportError:
    iio = None


# ============================================================================
# Video decoding — PyAV preferred, OpenCV fallback (matches analyze_phyworld.py)
# ============================================================================


def _decode_video(buf: bytes) -> Optional[np.ndarray]:
    """Decode MP4 bytes → ``(T, H, W, C) uint8``. None on failure."""
    # 1) PyAV — purely in-memory, no temp file
    if av is not None:
        try:
            container = av.open(io.BytesIO(buf))
            frames = [
                f.to_ndarray(format="rgb24")
                for f in container.decode(container.streams.video[0])
            ]
            container.close()
            if frames:
                return np.stack(frames)
        except Exception:
            pass

    # 2) OpenCV via temp file
    if cv2 is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                tf.write(buf)
                tf.flush()
                cap = cv2.VideoCapture(tf.name)
                frames = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame[..., ::-1])     # BGR → RGB
                cap.release()
            if frames:
                return np.stack(frames)
        except Exception:
            pass

    return None


# ============================================================================
# Dataset
# ============================================================================


class PhyWorldParabolaDataset(Dataset):
    """PhyWorld parabola evaluation set, decoded on the fly from the HDF5 file.

    Args:
        h5_path:         Path to ``parabola_eval.hdf5``.
        split:           ``"00000"`` (in-distribution, 1000 samples),
                         ``"00001"`` (out-of-distribution, 56 samples) or
                         ``"all"``.
        prompt_template: Format string for the text prompt of each sample. It
                         receives ``x0`` and ``v0``; the default template
                         states both values in the prompt.
        max_frames:      ``None`` keeps all frames, ``N`` subsamples uniformly
                         to N.
        image_size:      ``None`` keeps the native 256x256, otherwise the
                         frames are resized to this size.
    """

    DEFAULT_PROMPT = (
        "a small ball thrown into the air following a parabolic trajectory, "
        "initial horizontal position {x0:.2f} m, initial speed {v0:.2f} m/s"
    )

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "00000",
        prompt_template: Optional[str] = None,
        max_frames: Optional[int] = None,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.split = split
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.max_frames = max_frames
        self.image_size = image_size

        # Read static metadata once at init time so workers don't need to
        # re-traverse the file. The video bytes are loaded lazily per-worker.
        with h5py.File(self.h5_path, "r") as f:
            available = sorted(f["init_streams"].keys())
            if split == "all":
                self._splits = available
            elif split in available:
                self._splits = [split]
            else:
                raise KeyError(
                    f"split={split!r} not in {available}; pass 'all' for the union."
                )
            inits, positions, lens = [], [], []
            for s in self._splits:
                inits.append(f[f"init_streams/{s}"][:])
                positions.append(f[f"position_streams/{s}"][:])
                lens.append(f[f"video_streams/{s}"].shape[0])
        self.init_conditions = np.concatenate(inits, axis=0)        # (N, 2)
        self.positions = np.concatenate(positions, axis=0)          # (N, 32, 2)
        self._split_lens = lens
        self._cumlens = np.cumsum([0] + lens)
        self.length = sum(lens)

        # Lazy per-worker file handle (don't open here — h5py handles aren't
        # picklable across DataLoader fork/spawn).
        self._h5: Optional[h5py.File] = None

    # ── lazy file handle ────────────────────────────────────────────────────

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def _split_for(self, idx: int) -> tuple[str, int]:
        """Map a global idx to (split_name, in_split_idx)."""
        for s, start, end in zip(self._splits, self._cumlens, self._cumlens[1:]):
            if start <= idx < end:
                return s, idx - start
        raise IndexError(idx)

    # ── interface ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        f = self._file()
        split, local_idx = self._split_for(idx)

        # Decode video bytes
        buf = bytes(f[f"video_streams/{split}"][local_idx])
        frames = _decode_video(buf)
        if frames is None:
            raise RuntimeError(
                f"Failed to decode video for split={split} idx={local_idx}. "
                f"Install one of: pyav, imageio[pyav], opencv-python."
            )

        # Subsample frames if requested
        if self.max_frames is not None and frames.shape[0] != self.max_frames:
            t_idx = np.linspace(0, frames.shape[0] - 1, self.max_frames).astype(int)
            frames = frames[t_idx]

        # Optional bilinear resize
        if self.image_size is not None and (
            frames.shape[1] != self.image_size or frames.shape[2] != self.image_size
        ):
            t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()      # (T, C, H, W)
            t = torch.nn.functional.interpolate(
                t, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            frames = t.permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()

        imgs = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()  # (C, T, H, W) uint8

        x0, v0 = float(self.init_conditions[idx, 0]), float(self.init_conditions[idx, 1])
        prompt = self.prompt_template.format(x0=x0, v0=v0)

        return {
            "images": imgs,                                # (C, T, H, W) uint8
            "paths": f"phyworld_parabola/{split}/sample_{local_idx:05d}",
            "labels": 0,                                   # placeholder; no plausible/impossible
            "prompt": prompt,
            "init_conditions": torch.tensor([x0, v0], dtype=torch.float32),
            "trajectory": torch.from_numpy(self.positions[idx]).float(),  # (32, 2)
        }

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ============================================================================
# Collate — wraps collate_intphys and forwards the physics-condition fields
# ============================================================================


def collate_phyworld(samples, resolution_options, frame_count_options):
    """Same image / prompt / path / label collation as ``collate_intphys``,
    plus ``init_conditions`` (B, 2) and ``trajectory`` (B, 32, 2)."""
    base = collate_intphys(samples, resolution_options, frame_count_options)
    if base is None:
        return None
    base["init_conditions"] = torch.stack([s["init_conditions"] for s in samples])
    base["trajectory"]      = torch.stack([s["trajectory"]      for s in samples])
    return base


# ============================================================================
# Smoke test — `python -m data_pipeline.phyworld_dataset path/to/parabola_eval.hdf5`
# ============================================================================


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke test for PhyWorldParabolaDataset")
    ap.add_argument("h5_path", type=Path)
    ap.add_argument("--split", default="00000",
                    help="'00000', '00001', or 'all' (default: 00000).")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("phyworld_test"),
                    help="Where to dump preview PNGs.")
    args = ap.parse_args()

    if not args.h5_path.exists():
        sys.exit(f"file not found: {args.h5_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Construct + summarise
    print("=" * 70)
    print("Constructing PhyWorldParabolaDataset …")
    ds = PhyWorldParabolaDataset(
        h5_path=args.h5_path, split=args.split,
        max_frames=args.max_frames, image_size=args.image_size,
    )
    print(f"  len(ds)              = {len(ds)}")
    print(f"  init_conditions.shape= {ds.init_conditions.shape}")
    print(f"  positions.shape      = {ds.positions.shape}")
    print(f"  splits used          = {ds._splits}")
    print(f"  init range per dim   = "
          f"x0:[{ds.init_conditions[:,0].min():.2f}, {ds.init_conditions[:,0].max():.2f}]  "
          f"v0:[{ds.init_conditions[:,1].min():.2f}, {ds.init_conditions[:,1].max():.2f}]")

    # 2. Single-sample fetch
    print("\n" + "=" * 70)
    print("Single-sample fetch (idx=0) …")
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:18s} tensor shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:18s} {type(v).__name__} = {v!r}")
    imgs = s["images"]
    assert imgs.dtype == torch.uint8, "images must be uint8"
    assert imgs.dim() == 4 and imgs.shape[0] == 3, "images must be (C, T, H, W)"
    print(f"  images value-range = [{imgs.min().item()}, {imgs.max().item()}]")

    try:
        if iio is None:
            raise ImportError("imageio is not installed")
        first = imgs[:, 0].permute(1, 2, 0).numpy()
        last  = imgs[:, -1].permute(1, 2, 0).numpy()
        iio.imwrite(args.out_dir / "sample_000_first.png", first)
        iio.imwrite(args.out_dir / "sample_000_last.png",  last)
        print(f"  wrote first/last frame PNGs to {args.out_dir}")
    except Exception as e:
        print(f"  [WARN] couldn't save preview PNGs: {e}")

    # 3. DataLoader pass (verifies lazy h5 open under multiple workers)
    print("\n" + "=" * 70)
    print(f"DataLoader pass with {args.num_workers} workers, "
          f"batch={args.batch_size} …")
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
    )
    seen = 0
    for i, batch in enumerate(loader):
        ic = batch["init_conditions"]
        prompt0 = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
        print(f"  batch {i}: images={tuple(batch['images'].shape)}  "
              f"init={tuple(ic.shape)}  paths[0]={batch['paths'][0]}")
        print(f"            prompt[0] = {prompt0!r}")
        seen += batch["images"].shape[0]
        if seen >= args.n_samples:
            break

    print("\nAll good")
