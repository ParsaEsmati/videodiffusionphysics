#!/usr/bin/env python3
"""
CogVideoX driver: runs text-to-video generation, inversion of real videos with
per-block feature capture, or a VAE-only encode. Same datasets and output
layout as ``inference.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from diffusers.utils import export_to_video
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from data_pipeline.data_modules import IntPhysDataset, collate_intphys
from data_pipeline.inflevel_dataset import InfLevelDataset
from data_pipeline.phyworld_dataset import (
    PhyWorldParabolaDataset, collate_phyworld,
)
from models.integrators import INTEGRATORS
from models.cogvideox_pipeline import CogVideoXPipeline
from models.cogvideox_inversion import (
    CogInversionOutput,
    encode_video_to_latents_cogvideox,
    invert_cogvideox_t2v,
    prepare_video_for_inversion_cogvideox,
)


MODEL_INDEX_FILENAME = "model_index.json"
logger = logging.getLogger(__name__)


# ============================================================================
# Checkpoint + pipeline loading
# ============================================================================


def verify_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            "Download with:  hf download THUDM/CogVideoX-2b --local-dir <dir>"
        )
    if not (checkpoint_dir / MODEL_INDEX_FILENAME).is_file():
        raise FileNotFoundError(
            f"{MODEL_INDEX_FILENAME} not found under {checkpoint_dir}. "
            "The checkpoint may be incomplete."
        )
    logger.info(f"CogVideoX weights at {checkpoint_dir}")
    return checkpoint_dir


def load_cogvideox_pipeline(
    checkpoint_dir: Path,
    device: Optional[str] = None,
    dtype_name: str = "bfloat16",
) -> CogVideoXPipeline:
    weights_dir = verify_checkpoint(checkpoint_dir)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, dtype_name) if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading CogVideoXPipeline  dtype={dtype}  device={target_device}")
    pipeline = CogVideoXPipeline.from_pretrained(
        weights_dir, torch_dtype=dtype, local_files_only=True,
    ).to(target_device)
    logger.info(f"pipeline.scheduler type           = {type(pipeline.scheduler).__name__}")
    logger.info(
        "pipeline.scheduler.prediction_type = "
        f"{getattr(pipeline.scheduler.config, 'prediction_type', 'epsilon')}"
    )
    logger.info(
        f"pipeline.vae_scaling_factor_image  = {getattr(pipeline, 'vae_scaling_factor_image', None)}  "
        f"vs vae.config.scaling_factor = {pipeline.vae.config.scaling_factor}"
    )
    return pipeline


# ============================================================================
# Dataloader (same as inference.py)
# ============================================================================


DATASET_CLASSES = {
    "intphys":  IntPhysDataset,
    "inflevel": InfLevelDataset,
    "phyworld": PhyWorldParabolaDataset,
}


def build_dataloader(
    data_dir: str,
    meta_csv: str,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    max_frames: Optional[int] = None,
    dataset: str = "intphys",
    phyworld_split: str = "00000",
) -> DataLoader:
    if dataset == "phyworld":
        dataset_obj = PhyWorldParabolaDataset(
            h5_path=data_dir,
            split=phyworld_split,
            max_frames=None if max_frames is None else max_frames,
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
            max_frames=max_frames,
        )
        collate = partial(
            collate_intphys,
            resolution_options=[(width, height)],
            frame_count_options=[num_frames],
        )
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected one of {list(DATASET_CLASSES)}"
        )
    return DataLoader(
        dataset_obj,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )


# ============================================================================
# Generation (forward from prompts)
# ============================================================================


def run_generation(
    pipeline: CogVideoXPipeline,
    dataloader: DataLoader,
    output_dir: Path,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(dataloader.dataset)
    logger.info(
        f"CogVideoX T2V generation  samples={n_total}  {height}x{width}  "
        f"{num_frames}f  {fps}fps  steps={num_inference_steps}  cfg={guidance_scale}"
    )

    global_idx = 0
    for batch in dataloader:
        prompts = batch["prompts"]
        paths = batch["paths"]
        frames_list = pipeline(
            prompt=prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height, width=width, num_frames=num_frames,
        ).frames
        for path, frames in zip(paths, frames_list):
            stem = "_".join(Path(path).with_suffix("").parts[-3:])
            out_path = output_dir / f"{stem}.mp4"
            export_to_video(frames, str(out_path), fps=fps)
            global_idx += 1
            logger.info(f"  [{global_idx}/{n_total}] saved {out_path.name}")

    logger.info(f"Generation complete – {global_idx} video(s)")


# ============================================================================
# Reverse sampling (DDIM inversion)
# ============================================================================


def _tensor_to_pil_list(frames: torch.Tensor):
    """(C, T, H, W) float [0, 1] → list of PIL frames."""
    return [to_pil_image(frames[:, t].clamp(0, 1)) for t in range(frames.shape[1])]


def run_inversion(
    pipeline: CogVideoXPipeline,
    dataloader: DataLoader,
    output_dir: Path,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    reconstruct: bool = False,
    fps: int = 8,
    capture_steps: Optional[set] = None,
    capture_blocks: Optional[set] = None,
    integrator: str = "euler",
) -> None:
    """DDIM-invert each scene and save per-step block outputs + noise latent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(dataloader.dataset)
    logger.info(
        f"CogVideoX T2V inversion  samples={n_total}  {height}x{width}  "
        f"{num_frames}f  steps={num_inference_steps}  cfg={guidance_scale}  "
        f"integrator={integrator}"
    )

    def _save_step_batched(step_idx: int, step_data: dict, job_dirs: List[Path]) -> None:
        # Persist as (num_blocks, 1, D) — mean-pool the seq dim with keepdim so
        # files stay small and the probe (which does feat.mean(dim=1)) is unchanged.
        for b, job_dir in enumerate(job_dirs):
            step_dir = job_dir / f"step_{step_idx:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            block_tensor = torch.stack(
                [step_data[k][b] for k in sorted(step_data)]
            )
            block_tensor = block_tensor.mean(dim=1, keepdim=True).contiguous()
            torch.save(block_tensor, step_dir / "block_outputs.pt")

    do_cfg = guidance_scale > 1.0
    global_idx = 0
    for batch in dataloader:
        images = batch["images"]    # (B, C, T, H, W) float [0, 1]
        prompts = batch["prompts"]
        paths = batch["paths"]
        labels = batch["labels"]
        # PhyWorld-only: physics conditioning (None for IntPhys / InfLevel).
        init_conditions = batch.get("init_conditions")    # (B, 2) or None
        trajectories    = batch.get("trajectory")         # (B, 32, 2) or None
        B = images.shape[0]

        stems = ["_".join(Path(p).with_suffix("").parts[-3:]) for p in paths]
        job_dirs = []
        for stem in stems:
            jd = output_dir / stem
            jd.mkdir(parents=True, exist_ok=True)
            job_dirs.append(jd)

        logger.info(
            f"  [{global_idx + 1}–{global_idx + B}/{n_total}] "
            f"inverting batch of {B}: {', '.join(stems)}"
        )

        video_latents = torch.cat([
            prepare_video_for_inversion_cogvideox(pipeline, images[b], height, width)
            for b in range(B)
        ], dim=0)

        # DEBUG: encoded latent stats (should have std ~ 1 if VAE sampling is correct)
        vl = video_latents.flatten().float().cpu()
        logger.info(
            f"  video_latents shape={list(video_latents.shape)}  "
            f"mean={vl.mean().item():+.4f}  std={vl.std().item():.4f}  "
            f"min={vl.min().item():+.3f}  max={vl.max().item():+.3f}"
        )

        # CogVideoX is heavily text-conditioned. Empty prompts produce degenerate
        # velocity predictions during inversion (std drops well below 1.0), which
        # makes reconstruction collapse to a uniform / grayish output. Replace any
        # empty prompts with a generic placeholder so the model stays in-distribution.
        inv_prompts = [p if (p and p.strip()) else "a video of an object" for p in prompts]
        prompt_embeds, neg_embeds = pipeline.encode_prompt(
            prompt=inv_prompts,
            negative_prompt=[negative_prompt] * B if do_cfg else None,
            do_classifier_free_guidance=do_cfg,
            device=pipeline._execution_device,
        )

        step_callback = partial(_save_step_batched, job_dirs=job_dirs)

        result: CogInversionOutput = invert_cogvideox_t2v(
            pipeline=pipeline,
            video_latents=video_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds if do_cfg else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            capture_hidden_states=True,
            capture_steps=capture_steps,
            capture_blocks=capture_blocks,
            save_trajectory=False,
            step_callback=step_callback,
            integrator=integrator,
        )

        # DEBUG: confirm inverted noise is roughly Gaussian
        nl = result.noise_latent.flatten().float().cpu()
        logger.info(
            f"  noise_latent  shape={list(result.noise_latent.shape)}  "
            f"mean={nl.mean().item():+.4f}  std={nl.std().item():.4f}  "
            f"min={nl.min().item():+.3f}  max={nl.max().item():+.3f}"
        )

        for b, (stem, path, prompt, label) in enumerate(
            zip(stems, paths, prompts, labels.tolist())
        ):
            job_dir = job_dirs[b]
            torch.save(result.noise_latent[b:b+1].cpu(), job_dir / "noise_latent.pt")

            meta = {
                "prompt": prompt, "label": label, "path": path,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "integrator": integrator,
                "height": height, "width": width,
                "num_frames": num_frames, "mode": "T2V_cogvideox",
            }
            if init_conditions is not None:
                meta["init_conditions"] = init_conditions[b].tolist()
            if trajectories is not None:
                meta["trajectory"] = trajectories[b].tolist()

            if reconstruct:
                original_pil = _tensor_to_pil_list(images[b])
                export_to_video(original_pil, str(job_dir / "original.mp4"), fps=fps)
                noise_latent = result.noise_latent[b:b+1].to(pipeline.transformer.dtype)
                # CogVideoX needs text guidance to produce non-degenerate output;
                # use a generic prompt for reconstruction even though inversion was unconditional.
                recon_frames = pipeline(
                    prompt="a video of an object", latents=noise_latent,
                    negative_prompt=negative_prompt if do_cfg else None,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height, width=width, num_frames=num_frames,
                ).frames[0]
                export_to_video(recon_frames, str(job_dir / "reconstructed.mp4"), fps=fps)

            with open(job_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"    saved to {job_dir}/")

        global_idx += B

    logger.info(f"Inversion complete – {global_idx} sample(s)")


# ============================================================================
# VAE-only encode
# ============================================================================


def run_encoding(
    pipeline: CogVideoXPipeline,
    dataloader: DataLoader,
    output_dir: Path,
    height: int,
    width: int,
) -> None:
    """Encode videos with the CogVideoX VAE and save in probe-compatible format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(dataloader.dataset)
    logger.info(f"CogVideoX VAE-only encode  samples={n_total}  {height}x{width}")

    global_idx = 0
    for batch in dataloader:
        images = batch["images"]
        prompts = batch["prompts"]
        paths = batch["paths"]
        labels = batch["labels"]
        stems = ["_".join(Path(p).with_suffix("").parts[-3:]) for p in paths]

        for b, (stem, path, prompt, label) in enumerate(
            zip(stems, paths, prompts, labels.tolist())
        ):
            global_idx += 1
            job_dir = output_dir / stem
            step_dir = job_dir / "step_0000"
            step_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  [{global_idx}/{n_total}] encoding {stem}")

            latent = prepare_video_for_inversion_cogvideox(pipeline, images[b], height, width)
            # CogVideoX latent layout: (1, T, C, H, W)
            _, T_, C, H_, W_ = latent.shape
            # Reshape to (T*H*W, C) so probe sees `hidden_dim = C` consistently with WAN
            feat = latent.squeeze(0).permute(0, 2, 3, 1).reshape(T_ * H_ * W_, C)
            feat = feat.unsqueeze(0).cpu().float()   # (1, seq, C)
            torch.save(feat, step_dir / "block_outputs.pt")

            meta = {
                "prompt": prompt, "label": label, "path": path,
                "height": height, "width": width, "mode": "cogvideox_vae",
                "z_dim": C, "latent_shape": [T_, H_, W_],
            }
            with open(job_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    logger.info(f"Encoding complete – {global_idx} sample(s)")


# ============================================================================
# CLI
# ============================================================================


def parse_arguments():
    p = argparse.ArgumentParser(
        description="CogVideoX generation / DDIM inversion / VAE-encode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--ckpt-dir", type=Path, required=True,
                   help="CogVideoX model directory (contains model_index.json).")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Dataset root, or for --dataset phyworld the path to "
                        "parabola_eval.hdf5.")
    p.add_argument("--dataset", type=str, default="intphys", choices=list(DATASET_CLASSES))
    p.add_argument("--meta-csv", type=Path, default=None,
                   help="Required for intphys / inflevel; ignored for phyworld.")
    p.add_argument("--phyworld-split", type=str, default="00000",
                   help="PhyWorld split: '00000' (1000 id), '00001' (56 ood), 'all'.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))

    # Modes
    p.add_argument("--reverse_sample", action="store_true",
                   help="DDIM-invert real videos instead of forward generation.")
    p.add_argument("--encode-only", action="store_true",
                   help="VAE-only baseline (no diffusion).")
    p.add_argument("--reconstruct", action="store_true",
                   help="After inversion, re-generate from inverted noise and save "
                        "original.mp4 + reconstructed.mp4. Use with --reverse_sample.")

    # Runtime
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=6.0,
                   help="CFG scale (default 6.0 for CogVideoX; use 1.0 for exact inversion).")
    p.add_argument("--integrator", type=str, default="euler", choices=list(INTEGRATORS),
                   help="Inversion integrator: euler makes one model call per step, heun "
                        "makes two and averages them (default: euler).")
    p.add_argument("--negative-prompt", type=str, default="")

    # Shape — CogVideoX defaults: 480×720×49
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=49,
                   help="CogVideoX wants (num_frames - 1) divisible by 4; default 49.")
    p.add_argument("--fps", type=int, default=8)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    # Capture
    p.add_argument("--capture-steps", nargs="*", type=int, default=None, metavar="STEP",
                   help="Inversion step indices to capture (0-indexed). Default: all.")
    p.add_argument("--capture-blocks", nargs="*", type=int, default=None, metavar="BLOCK",
                   help="Transformer block indices to capture. Default: all.")
    p.add_argument("--save-every", type=int, default=None, metavar="N",
                   help="Capture every N-th step (ignored if --capture-steps given).")

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

    pipeline = load_cogvideox_pipeline(args.ckpt_dir, args.device, args.dtype)

    if args.reverse_sample or args.encode_only:
        max_frames = None
    else:
        max_frames = 0   # prompts-only generation

    dataloader = build_dataloader(
        data_dir=args.data_dir,
        meta_csv=str(args.meta_csv) if args.meta_csv is not None else "",
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_frames=max_frames, dataset=args.dataset,
        phyworld_split=args.phyworld_split,
    )

    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.encode_only:
        run_encoding(
            pipeline=pipeline, dataloader=dataloader,
            output_dir=run_dir, height=args.height, width=args.width,
        )
    elif args.reverse_sample:
        capture_steps = set()
        if args.capture_steps is not None:
            capture_steps.update(args.capture_steps)
        if args.save_every is not None:
            capture_steps.update(range(0, args.steps, args.save_every))
        if not capture_steps:
            capture_steps = None

        capture_blocks = set(args.capture_blocks) if args.capture_blocks is not None else None

        run_inversion(
            pipeline=pipeline, dataloader=dataloader,
            output_dir=run_dir, negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            height=args.height, width=args.width, num_frames=args.num_frames,
            reconstruct=args.reconstruct, fps=args.fps,
            capture_steps=capture_steps, capture_blocks=capture_blocks,
            integrator=args.integrator,
        )
    else:
        run_generation(
            pipeline=pipeline, dataloader=dataloader,
            output_dir=run_dir, negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            height=args.height, width=args.width, num_frames=args.num_frames,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
