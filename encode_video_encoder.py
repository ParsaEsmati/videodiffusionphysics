#!/usr/bin/env python3
"""
Baseline encoders: runs a Hugging Face video encoder (V-JEPA 2, VideoMAE) over
a dataset and saves the output of every block in the layout the probes read.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from data_pipeline.data_modules import IntPhysDataset, collate_intphys
from data_pipeline.inflevel_dataset import InfLevelDataset
from data_pipeline.phyworld_dataset import (
    PhyWorldParabolaDataset, collate_phyworld,
)
from models.block_capture import (
    capture_forward_hooks,
    find_transformer_blocks,
    get_blocks_by_path,
)


logger = logging.getLogger(__name__)


DATASET_CLASSES = {
    "intphys": IntPhysDataset,
    "inflevel": InfLevelDataset,
    "phyworld": PhyWorldParabolaDataset,
}

# ImageNet stats — standard for ViT-based video encoders
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# ============================================================================
# Model loading
# ============================================================================


def load_encoder(ckpt: str, device: str, dtype: torch.dtype):
    """Load any HuggingFace video encoder via ``AutoModel``."""
    logger.info(f"Loading encoder from {ckpt}  dtype={dtype}  device={device}")
    model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ============================================================================
# Video preprocessing
# ============================================================================


def preprocess_video(
    video: torch.Tensor,       # (C, T, H, W) in [0, 1]
    num_frames: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Subsample to ``num_frames``, resize to ``image_size`` and ImageNet-normalize.
    Returns a ``(1, C, T, H, W)`` tensor."""
    C, T, H, W = video.shape

    if T != num_frames:
        idx = torch.linspace(0, T - 1, num_frames).long()
        video = video[:, idx]

    if (H, W) != (image_size, image_size):
        video = video.permute(1, 0, 2, 3).contiguous()   # (T, C, H, W)
        video = F.interpolate(
            video, size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        )
        video = video.permute(1, 0, 2, 3).contiguous()   # back to (C, T, H, W)

    mean = IMAGENET_MEAN.to(device=video.device, dtype=video.dtype)
    std = IMAGENET_STD.to(device=video.device, dtype=video.dtype)
    video = (video - mean) / std

    return video.unsqueeze(0).to(device=device, dtype=dtype)   # (1, C, T, H, W)


# ============================================================================
# Encoding loop
# ============================================================================


def run_encoding(
    model,
    dataloader: DataLoader,
    output_dir: Path,
    num_frames: int,
    image_size: int,
    blocks_path: Optional[str],
    input_layout: str,
) -> None:
    """Run the encoder on each scene and save per-block activations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if blocks_path:
        blocks = get_blocks_by_path(model, blocks_path)
        path_used = blocks_path
    else:
        blocks, path_used = find_transformer_blocks(model)
    logger.info(f"Hooking {len(blocks)} transformer blocks at `{path_used}`")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_total = len(dataloader.dataset)
    global_idx = 0

    for batch in dataloader:
        images = batch["images"]     # (B, C, T, H, W) in [0, 1]
        prompts = batch["prompts"]
        paths = batch["paths"]
        labels = batch["labels"]
        # PhyWorld-only: physics conditioning (None for IntPhys / InfLevel).
        init_conditions = batch.get("init_conditions")    # (B, 2) or None
        trajectories    = batch.get("trajectory")         # (B, 32, 2) or None
        stems = ["_".join(Path(p).with_suffix("").parts[-3:]) for p in paths]

        for b, (stem, path, prompt, label) in enumerate(
            zip(stems, paths, prompts, labels.tolist())
        ):
            global_idx += 1
            job_dir = output_dir / stem
            step_dir = job_dir / "step_0000"
            step_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  [{global_idx}/{n_total}] encoding {stem}")

            video = preprocess_video(
                images[b], num_frames=num_frames, image_size=image_size,
                device=device, dtype=dtype,
            )   # (1, C, T, H, W)

            if input_layout == "BTCHW":
                video_input = video.permute(0, 2, 1, 3, 4).contiguous()
            else:
                video_input = video

            with capture_forward_hooks(blocks) as captured:
                with torch.no_grad():
                    outputs = model(video_input)

            if len(captured) != len(blocks):
                raise RuntimeError(
                    f"Captured {len(captured)}/{len(blocks)} blocks — model forward "
                    f"may not have reached every block. Check --input-layout "
                    f"(tried {input_layout}) and --blocks-path."
                )

            # Append the model's official final output (last_hidden_state, after any
            # final layernorm/pooling-projection in the encoder). This is the
            # "encoded latent" you'd actually pass to a downstream task. In some HF
            # video encoders, last_hidden_state ≈ blocks[-1].output passed through one
            # additional layernorm — the difference is small but real.
            if hasattr(outputs, "last_hidden_state"):
                final = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                final = outputs[0]
            else:
                final = outputs
            captured[len(blocks)] = final.detach().float().cpu()

            # Each captured[i]: (1, seq, D)  →  stack to (num_blocks + 1, seq, D),
            # then mean-pool the seq dim with keepdim so the persisted file is
            # (num_blocks + 1, 1, D) — matches the diffusion pipelines.
            feat = torch.stack([captured[i].squeeze(0) for i in sorted(captured)], dim=0)
            feat = feat.mean(dim=1, keepdim=True).contiguous()

            torch.save(feat, step_dir / "block_outputs.pt")

            meta = {
                "prompt": prompt, "label": label, "path": path,
                "image_size": image_size, "num_frames": num_frames,
                "encoder": "auto", "pool": "mean_all",
                "blocks_path": path_used,
                "num_blocks": feat.shape[0],
                "num_transformer_blocks": len(blocks),
                "final_output_index": feat.shape[0] - 1,
                "seq": feat.shape[1],
                "hidden_dim": feat.shape[2],
            }
            if init_conditions is not None:
                meta["init_conditions"] = init_conditions[b].tolist()
            if trajectories is not None:
                meta["trajectory"] = trajectories[b].tolist()
            with open(job_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    logger.info(f"Encoding complete – {global_idx} sample(s)")


# ============================================================================
# Dataloader
# ============================================================================


def build_dataloader(
    data_dir: str, meta_csv: str,
    height: int, width: int, num_frames: int,
    dataset: str, batch_size: int, num_workers: int,
    phyworld_split: str = "00000",
) -> DataLoader:
    if dataset == "phyworld":
        dataset_obj = PhyWorldParabolaDataset(
            h5_path=data_dir,
            split=phyworld_split,
            max_frames=None,
            image_size=None,
        )
        collate = partial(
            collate_phyworld,
            resolution_options=[(width, height)],
            frame_count_options=[num_frames],
        )
    elif dataset in DATASET_CLASSES:
        dataset_cls = DATASET_CLASSES[dataset]
        dataset_obj = dataset_cls(
            data_dir=data_dir,
            meta_path=meta_csv,
            skip_missing_files=True,
            resolution_options=[(width, height)],
            max_frames=None,
        )
        collate = partial(
            collate_intphys,
            resolution_options=[(width, height)],
            frame_count_options=[num_frames],
        )
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected {list(DATASET_CLASSES)}"
        )
    return DataLoader(
        dataset_obj,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )


# ============================================================================
# CLI
# ============================================================================


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Per-block feature extraction from any HF video encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="HuggingFace model id or local path (V-JEPA, VideoMAE, ...).")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Dataset root, or for --dataset phyworld the path to "
                        "parabola_eval.hdf5.")
    p.add_argument("--dataset", type=str, default="inflevel", choices=list(DATASET_CLASSES))
    p.add_argument("--meta-csv", type=str, default=None,
                   help="Required for intphys / inflevel; ignored for phyworld.")
    p.add_argument("--phyworld-split", type=str, default="00000",
                   help="PhyWorld split: '00000' (1000 id), '00001' (56 ood), 'all'.")
    p.add_argument("--output-dir", type=Path, required=True)

    # Model-specific input configuration — match the checkpoint's expected resolution/frames
    p.add_argument("--num-frames", type=int, default=16,
                   help="Frames fed to the encoder (V-JEPA 2: 64, VideoMAE v2: 16).")
    p.add_argument("--image-size", type=int, default=224,
                   help="Spatial size (V-JEPA 2: 256, VideoMAE v2: 224).")
    p.add_argument("--input-layout", type=str, default="BCTHW", choices=["BCTHW", "BTCHW"],
                   help="Model forward's expected tensor layout. V-JEPA 2: BCTHW. "
                        "VideoMAE: BTCHW.")
    p.add_argument("--blocks-path", type=str, default=None,
                   help="Explicit dotted path to the transformer block ModuleList "
                        "(e.g. 'encoder.layer'). Autodetected if omitted.")
    # NOTE: --pool was removed — features are always mean-pooled to
    # (num_blocks, 1, D) before saving, matching the diffusion pipelines.

    # Loader
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    # Runtime
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dtype = getattr(torch, args.dtype)
    model = load_encoder(args.ckpt, args.device, dtype=dtype)

    dataloader = build_dataloader(
        data_dir=args.data_dir,
        meta_csv=args.meta_csv if args.meta_csv is not None else "",
        height=args.image_size, width=args.image_size,
        num_frames=args.num_frames,
        dataset=args.dataset,
        batch_size=args.batch_size, num_workers=args.num_workers,
        phyworld_split=args.phyworld_split,
    )

    run_encoding(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        blocks_path=args.blocks_path,
        input_layout=args.input_layout,
    )


if __name__ == "__main__":
    main()
