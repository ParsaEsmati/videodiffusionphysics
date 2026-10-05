#!/usr/bin/env python3
"""
WAN driver: runs text-to-video or image-to-video generation, inversion of real
videos with per-block feature capture, or a VAE-only encode, over IntPhys,
InfLevel or PhyWorld.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.io as tvio
from diffusers import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPVisionModel

from data_pipeline.data_modules import IntPhysDataset, collate_intphys
from data_pipeline.inflevel_dataset import InfLevelDataset
from data_pipeline.phyworld_dataset import (
    PhyWorldParabolaDataset, collate_phyworld,
)
from models import WanPipeline, WanImageToVideoPipeline
from models.integrators import INTEGRATORS
from models.wan_inversion import (
    InversionOutput,
    invert_t2v,
    invert_i2v,
    manual_generate_t2v,
    decode_latents_to_video,
    prepare_video_for_inversion,
    prepare_i2v_condition,
)


MODEL_INDEX_FILENAME = "model_index.json"
logger = logging.getLogger(__name__)


# ── DEBUG: precision-floor diagnostics for reconstruction. Remove before release. ──
# Toggle individually to attribute round-trip distortion to each noise source.
_DEBUG_FP32_NOISE = False         # Pass inverted noise to pipeline as fp32 (skip fp32→bf16 cast).
_DEBUG_DETERMINISTIC_ATTN = False  # Force SDPA math backend during reconstruction (no FlashAttention).
_DEBUG_FP32_TRANSFORMER = False   # Override dtype to fp32 when loading the pipeline.
_DEBUG_FORCE_EULER_SCHEDULER = False  # Replace pipeline scheduler with FlowMatchEuler (matches inversion's Euler).
_DEBUG_PERTURB_BEFORE_INVERSION = False  # Add sigma_min·randn to clean latent before inverting (bridges sigma=0 OOD at step 0).


def _maybe_deterministic_attn_ctx():
    """Context manager for reconstruction pipeline call (DEBUG)."""
    if _DEBUG_DETERMINISTIC_ATTN:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False,
        )
    return contextlib.nullcontext()


def _maybe_perturb_clean_latent(
    latents: torch.Tensor,
    scheduler,
    num_inference_steps: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """If the debug flag is on, add sigma_min-scaled noise to the clean latent, so that
    inversion step 0 sees an input at a noise level the model was trained on."""
    if not _DEBUG_PERTURB_BEFORE_INVERSION:
        return latents
    scheduler.set_timesteps(num_inference_steps, device=device)
    sigma_min = scheduler.sigmas[-2].item()   # sigmas[-1] is 0, [-2] is smallest non-zero
    if seed is not None:
        gen = torch.Generator(device=latents.device).manual_seed(seed)
        eps = torch.randn(latents.shape, generator=gen, device=latents.device, dtype=latents.dtype)
    else:
        eps = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
    logger.info(f"DEBUG: perturbing clean latent with sigma_min={sigma_min:.6f} * randn")
    return latents + sigma_min * eps


def resolve_capture_steps_from_fractions(
    scheduler,
    num_inference_steps: int,
    fractions: List[float],
    device: torch.device,
) -> set:
    """Map noise-level fractions (0 = clean, 1 = max noise) to the nearest inversion step indices."""
    scheduler.set_timesteps(num_inference_steps, device=device)
    # Inversion sigmas: ascending from 0 → sigma_max
    inv_sigmas = scheduler.sigmas.flip(0)  # [0, sigma_min, ..., sigma_max]
    sigma_max = inv_sigmas[-1]

    capture = set()
    for frac in fractions:
        target_sigma = frac * sigma_max
        # inv_sigmas has N+1 entries (one per boundary); step indices are 0..N-1
        # Find the closest sigma among step boundaries, clamped to valid step range
        idx = torch.argmin((inv_sigmas - target_sigma).abs()).item()
        step = min(idx, num_inference_steps - 1)
        capture.add(step)

    logger.info(
        f"  capture-fractions {fractions} → step indices {sorted(capture)} "
        f"(sigma_max={sigma_max:.4f})"
    )
    return capture


# ============================================================================
# Checkpoint utilities
# ============================================================================


def _find_model_index(search_dir: Path) -> Optional[Path]:
    if not search_dir.exists():
        return None
    if (search_dir / MODEL_INDEX_FILENAME).is_file():
        return search_dir
    for candidate in search_dir.rglob(MODEL_INDEX_FILENAME):
        return candidate.parent
    return None


def verify_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the model root path, raising FileNotFoundError if missing."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            "Download with:  hf download Wan-AI/Wan2.1-... --local-dir <dir>"
        )
    model_root = _find_model_index(checkpoint_dir)
    if model_root is None:
        raise FileNotFoundError(
            f"{MODEL_INDEX_FILENAME} not found under {checkpoint_dir}. "
            "The checkpoint may be incomplete."
        )
    logger.info(f"Model weights at {model_root}")
    return model_root


# ============================================================================
# Pipeline loading
# ============================================================================


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    detected = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Auto-detected device: {detected}")
    return detected


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name) if torch.cuda.is_available() else torch.float32


def load_t2v_pipeline(
    checkpoint_dir: Path,
    device: Optional[str] = None,
    dtype_name: str = "bfloat16",
) -> WanPipeline:
    weights_dir = verify_checkpoint(checkpoint_dir)
    target_device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name)

    logger.info(f"Loading T2V pipeline  dtype={dtype}  device={target_device}")
    pipeline = WanPipeline.from_pretrained(
        weights_dir, local_files_only=True, torch_dtype=dtype
    ).to(target_device)
    logger.info("T2V pipeline ready")
    return pipeline


def load_i2v_pipeline(
    checkpoint_dir: Path,
    device: Optional[str] = None,
    dtype_name: str = "bfloat16",
) -> WanImageToVideoPipeline:
    weights_dir = verify_checkpoint(checkpoint_dir)
    target_device = _resolve_device(device)
    dtype = _resolve_dtype(dtype_name)

    logger.info(f"Loading I2V pipeline  dtype={dtype}  device={target_device}")

    # VAE and image encoder stay in float32 for numerical stability
    vae = AutoencoderKLWan.from_pretrained(
        weights_dir, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        weights_dir, subfolder="image_encoder", torch_dtype=torch.float32, local_files_only=True
    )
    pipeline = WanImageToVideoPipeline.from_pretrained(
        weights_dir, vae=vae, image_encoder=image_encoder,
        local_files_only=True, torch_dtype=dtype,
    ).to(target_device)
    logger.info("I2V pipeline ready")
    return pipeline


# ============================================================================
# Dataloader
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
    """Build a DataLoader for IntPhys, InfLevel or PhyWorld (``data_dir`` is the HDF5 file for PhyWorld).
    ``max_frames``: ``None`` loads all frames, ``1`` the first frame only, ``0`` none (prompts only)."""
    if dataset == "phyworld":
        dataset_obj = PhyWorldParabolaDataset(
            h5_path=data_dir,
            split=phyworld_split,
            max_frames=None if max_frames is None else max_frames,
            image_size=None,                # let collate handle resize
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )


# ============================================================================
# Video file loader (kept for future .mp4 support)
# ============================================================================


def load_video_as_frames(video_path: Path) -> List[Image.Image]:
    """Decode a video file and return its frames as a list of PIL Images."""
    frames_tensor, _, _ = tvio.read_video(str(video_path), output_format="TCHW", pts_unit="sec")
    to_pil = transforms.ToPILImage()
    return [to_pil(frame) for frame in frames_tensor]


# ============================================================================
# Forward generation
# ============================================================================


def run_generation(
    pipeline: Union[WanPipeline, WanImageToVideoPipeline],
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
    """Generate videos from noise using prompts delivered by the dataloader."""
    is_i2v = isinstance(pipeline, WanImageToVideoPipeline)
    mode = "I2V" if is_i2v else "T2V"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(dataloader.dataset)
    logger.info(
        f"WAN {mode} generation  samples={n_total}  {height}x{width}  "
        f"{num_frames}f  {fps}fps  steps={num_inference_steps}  cfg={guidance_scale}"
    )

    global_idx = 0
    for batch in dataloader:
        prompts = batch["prompts"]
        paths = batch["paths"]

        if is_i2v:
            # I2V pipeline accepts only a single image — iterate per sample
            first_frames = _first_frames_as_pil(batch["images"])
            for path, prompt, first_frame in zip(paths, prompts, first_frames):
                frames = pipeline(
                    prompt=prompt,
                    image=first_frame,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                ).frames[0]
                stem = "_".join(Path(path).with_suffix("").parts[-3:])
                out_path = output_dir / f"{stem}.mp4"
                export_to_video(frames, str(out_path), fps=fps)
                global_idx += 1
                logger.info(f"  [{global_idx}/{n_total}] saved {out_path.name}")
        else:
            frames_list = pipeline(
                prompt=prompts,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_frames=num_frames,
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


def _tensor_to_pil_list(frames: torch.Tensor) -> List[Image.Image]:
    """Convert a ``(C, T, H, W)`` float [0, 1] tensor to a list of PIL Images."""
    return [to_pil_image(frames[:, t].clamp(0, 1)) for t in range(frames.shape[1])]


def _first_frames_as_pil(images: torch.Tensor) -> List[Image.Image]:
    """First frame of each video in a ``(B, C, T, H, W)`` float [0, 1] batch, as PIL images."""
    return [to_pil_image(images[b, :, 0].clamp(0, 1)) for b in range(images.shape[0])]


def run_inversion(
    pipeline: Union[WanPipeline, WanImageToVideoPipeline],
    dataloader: DataLoader,
    output_dir: Path,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    reconstruct: bool = False,
    fps: int = 16,
    temporal_pool: bool = False,
    capture_steps: Optional[set] = None,
    capture_blocks: Optional[set] = None,
    integrator: str = "euler",
) -> None:
    """Invert each scene and save ``step_NNNN/block_outputs.pt``, ``noise_latent.pt`` and ``meta.json``
    in its folder (plus ``original.mp4`` and ``reconstructed.mp4`` with ``reconstruct``)."""
    is_i2v = isinstance(pipeline, WanImageToVideoPipeline)
    mode = "I2V" if is_i2v else "T2V"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(dataloader.dataset)
    logger.info(
        f"WAN {mode} inversion  samples={n_total}  {height}x{width}  "
        f"{num_frames}f  steps={num_inference_steps}  cfg={guidance_scale}  "
        f"integrator={integrator}"
    )

    def _save_step_batched(step_idx: int, step_data: dict, job_dirs: List[Path]) -> None:
        """Save one captured step per sample as ``(num_blocks, 1, D)``, mean-pooled over tokens
        (the probe averages over tokens anyway)."""
        for b, job_dir in enumerate(job_dirs):
            step_dir = job_dir / f"step_{step_idx:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            block_tensor = torch.stack(
                [step_data[k][b] for k in sorted(step_data)]
            )
            block_tensor = block_tensor.mean(dim=1, keepdim=True).contiguous()
            torch.save(block_tensor, step_dir / "block_outputs.pt")
        logger.debug(f"      saved step {step_idx} for {len(job_dirs)} samples")

    do_cfg = guidance_scale > 1.0
    global_idx = 0
    for batch in dataloader:
        images = batch["images"]    # (B, C, T, H, W) float [0, 1]
        prompts = batch["prompts"]  # List[str]
        paths = batch["paths"]      # List[str]
        labels = batch["labels"]    # (B,)
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

        if is_i2v:
            # I2V: prepare_i2v_condition hardcodes batch_size=1, process individually
            for b, (path, prompt, label) in enumerate(zip(paths, prompts, labels.tolist())):
                frames = images[b]
                job_dir = job_dirs[b]
                global_idx += 1

                video_latents = prepare_video_for_inversion(pipeline, frames, height, width)
                video_latents = _maybe_perturb_clean_latent(
                    video_latents, pipeline.scheduler, num_inference_steps,
                    pipeline._execution_device, seed=global_idx,
                )
                prompt_embeds, neg_embeds = pipeline.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt if do_cfg else None,
                    do_classifier_free_guidance=do_cfg,
                    device=pipeline._execution_device,
                )
                step_callback = partial(_save_step_batched, job_dirs=[job_dir])

                first_frame = _tensor_to_pil_list(frames)[0]
                condition, image_embeds = prepare_i2v_condition(
                    pipeline, first_frame, height, width, num_frames
                )
                result: InversionOutput = invert_i2v(
                    pipeline=pipeline,
                    video_latents=video_latents,
                    condition=condition,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds if do_cfg else None,
                    image_embeds=image_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    capture_hidden_states=True,
                    temporal_pool=temporal_pool,
                    capture_steps=capture_steps,
                    capture_blocks=capture_blocks,
                    save_trajectory=False,
                    step_callback=step_callback,
                )

                torch.save(result.noise_latent.cpu(), job_dir / "noise_latent.pt")
                _print_latent_stats(f"noise_latent[{stems[b]}]", result.noise_latent)
                meta = {
                    "prompt": prompt, "label": label, "path": path,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "height": height, "width": width,
                    "num_frames": num_frames, "mode": mode,
                }

                if reconstruct:
                    original_pil = _tensor_to_pil_list(frames)
                    export_to_video(original_pil, str(job_dir / "original.mp4"), fps=fps)
                    noise_dtype = torch.float32 if _DEBUG_FP32_NOISE else pipeline.transformer.dtype
                    noise_latent = result.noise_latent.to(noise_dtype)
                    with _maybe_deterministic_attn_ctx():
                        recon_frames = pipeline(
                            prompt=prompt, image=first_frame, latents=noise_latent,
                            negative_prompt=negative_prompt if do_cfg else None,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=height, width=width, num_frames=num_frames,
                        ).frames[0]
                    export_to_video(recon_frames, str(job_dir / "reconstructed.mp4"), fps=fps)

                with open(job_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info(f"    saved to {job_dir}/")
        else:
            # T2V: batch all samples through a single inversion call
            video_latents = torch.cat([
                prepare_video_for_inversion(pipeline, images[b], height, width)
                for b in range(B)
            ], dim=0)  # (B, z_dim, T', H', W')
            video_latents = _maybe_perturb_clean_latent(
                video_latents, pipeline.scheduler, num_inference_steps,
                pipeline._execution_device, seed=global_idx,
            )

            prompt_embeds, neg_embeds = pipeline.encode_prompt(
                prompt=prompts,
                negative_prompt=[negative_prompt] * B if do_cfg else None,
                do_classifier_free_guidance=do_cfg,
                device=pipeline._execution_device,
            )

            step_callback = partial(_save_step_batched, job_dirs=job_dirs)

            result: InversionOutput = invert_t2v(
                pipeline=pipeline,
                video_latents=video_latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds if do_cfg else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                capture_hidden_states=True,
                temporal_pool=temporal_pool,
                capture_steps=capture_steps,
                capture_blocks=capture_blocks,
                save_trajectory=False,
                step_callback=step_callback,
                integrator=integrator,
            )

            # Split results per sample and save
            for b, (stem, path, prompt, label) in enumerate(
                zip(stems, paths, prompts, labels.tolist())
            ):
                job_dir = job_dirs[b]
                torch.save(result.noise_latent[b:b+1].cpu(), job_dir / "noise_latent.pt")
                _print_latent_stats(f"noise_latent[{stem}]", result.noise_latent[b:b+1])

                meta = {
                    "prompt": prompt, "label": label, "path": path,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "integrator": integrator,
                    "height": height, "width": width,
                    "num_frames": num_frames, "mode": mode,
                }
                if init_conditions is not None:
                    meta["init_conditions"] = init_conditions[b].tolist()
                if trajectories is not None:
                    meta["trajectory"] = trajectories[b].tolist()

                if reconstruct:
                    original_pil = _tensor_to_pil_list(images[b])
                    export_to_video(original_pil, str(job_dir / "original.mp4"), fps=fps)
                    noise_dtype = torch.float32 if _DEBUG_FP32_NOISE else pipeline.transformer.dtype
                    noise_latent = result.noise_latent[b:b+1].to(noise_dtype)
                    with _maybe_deterministic_attn_ctx():
                        recon_frames = pipeline(
                            prompt=prompt, latents=noise_latent,
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
# Inversion diagnostic: forward → invert → forward (skips VAE)
# ============================================================================


def _print_latent_stats(label: str, t: torch.Tensor) -> None:
    """One-line summary of a latent tensor's distribution."""
    x = t.flatten().float().cpu()
    logger.info(
        f"  {label:22s} shape={list(t.shape)}  "
        f"mean={x.mean().item():+.4f}  std={x.std().item():.4f}  "
        f"min={x.min().item():+.4f}  max={x.max().item():+.4f}"
    )


def run_inversion_diagnostic(
    pipeline: WanPipeline,
    num_inference_steps: int,
    height: int,
    width: int,
    num_frames: int,
    seed: int = 0,
    integrator: str = "euler",
) -> None:
    """Round-trip check without the VAE: denoise fixed-seed noise, invert the result, denoise
    again, and report the drift in the noise and in the clean latent."""
    device = pipeline._execution_device

    logger.info(
        f"Inversion diagnostic: N={num_inference_steps}  {height}x{width}  "
        f"{num_frames}f  seed={seed}"
    )

    num_latent_frames = (num_frames - 1) // pipeline.vae_scale_factor_temporal + 1
    latent_shape = (
        1,
        pipeline.transformer.config.in_channels,
        num_latent_frames,
        height // pipeline.vae_scale_factor_spatial,
        width // pipeline.vae_scale_factor_spatial,
    )

    # 1) Fresh noise
    gen = torch.Generator(device=device).manual_seed(seed)
    z_noise_init = torch.randn(
        latent_shape, generator=gen, device=device, dtype=torch.float32,
    )
    _print_latent_stats("z_noise_init", z_noise_init)

    # 2) Forward denoise → clean latent (output_type="latent" skips VAE decode)
    logger.info("  [1/3] forward denoise: noise → clean")
    with _maybe_deterministic_attn_ctx():
        z_clean = pipeline(
            prompt="", latents=z_noise_init.clone(),
            num_inference_steps=num_inference_steps, guidance_scale=1.0,
            height=height, width=width, num_frames=num_frames,
            output_type="latent",
        ).frames
    _print_latent_stats("z_clean", z_clean)

    # 3) Invert clean latent back to noise
    logger.info(f"  [2/3] inversion:       clean → noise  (integrator={integrator})")
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt="", do_classifier_free_guidance=False, device=device,
    )
    z_clean_for_inv = _maybe_perturb_clean_latent(
        z_clean.to(torch.float32), pipeline.scheduler, num_inference_steps, device, seed=seed,
    )
    result = invert_t2v(
        pipeline=pipeline,
        video_latents=z_clean_for_inv,
        prompt_embeds=prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        capture_hidden_states=False,
        save_trajectory=False,
        integrator=integrator,
    )
    z_noise_recovered = result.noise_latent
    _print_latent_stats("z_noise_recovered", z_noise_recovered)

    # 4) Forward denoise again, from recovered noise
    logger.info("  [3/3] forward denoise: recovered noise → clean (round-trip)")
    noise_in = z_noise_recovered.to(torch.float32) if _DEBUG_FP32_NOISE else z_noise_recovered.to(pipeline.transformer.dtype)
    with _maybe_deterministic_attn_ctx():
        z_clean_rt = pipeline(
            prompt="", latents=noise_in,
            num_inference_steps=num_inference_steps, guidance_scale=1.0,
            height=height, width=width, num_frames=num_frames,
            output_type="latent",
        ).frames
    _print_latent_stats("z_clean_rt", z_clean_rt)

    # 5) Compare
    z_noise_init_f = z_noise_init.float()
    z_noise_recovered_f = z_noise_recovered.float().to(z_noise_init_f.device)
    z_clean_f = z_clean.float()
    z_clean_rt_f = z_clean_rt.float().to(z_clean_f.device)

    noise_l2 = (z_noise_init_f - z_noise_recovered_f).pow(2).mean().sqrt()
    clean_l2 = (z_clean_f - z_clean_rt_f).pow(2).mean().sqrt()
    noise_rel = noise_l2 / z_noise_init_f.std()
    clean_rel = clean_l2 / z_clean_f.std()

    logger.info(
        f"  noise round-trip  RMSE={noise_l2.item():.4f}  "
        f"(relative to std: {noise_rel.item():.4f})"
    )
    logger.info(
        f"  clean round-trip  RMSE={clean_l2.item():.4f}  "
        f"(relative to std: {clean_rel.item():.4f})"
    )


# ============================================================================
# VAE-only encode (baseline for probe comparison)
# ============================================================================


def run_encoding(
    pipeline: Union[WanPipeline, WanImageToVideoPipeline],
    dataloader: DataLoader,
    output_dir: Path,
    height: int,
    width: int,
) -> None:
    """VAE-only baseline: save each video's latents as ``step_0000/block_outputs.pt`` with shape
    ``(1, tokens, z_dim)``, in the layout the probe reads (``--num-blocks 1 --hidden-dim <z_dim>``)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(dataloader.dataset)
    logger.info(f"WAN VAE-only encode  samples={n_total}  {height}x{width}")

    global_idx = 0
    for batch in dataloader:
        images = batch["images"]    # (B, C, T, H, W) float [0, 1]
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

            latent = prepare_video_for_inversion(pipeline, images[b], height, width)
            # latent: (1, z_dim, T', H', W') → (1, T'*H'*W', z_dim)
            _, C, T_, H_, W_ = latent.shape
            feat = latent.squeeze(0).permute(1, 2, 3, 0).reshape(T_ * H_ * W_, C)
            feat = feat.unsqueeze(0).cpu().float()   # (num_blocks=1, seq, D)
            torch.save(feat, step_dir / "block_outputs.pt")

            meta = {
                "prompt": prompt, "label": label, "path": path,
                "height": height, "width": width, "mode": "VAE",
                "z_dim": C, "latent_shape": [T_, H_, W_],
            }
            with open(job_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    logger.info(f"Encoding complete – {global_idx} sample(s)")


# ============================================================================
# CLI
# ============================================================================


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WAN batch video generation and inversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--ckpt-dir", type=Path, required=True,
        help="Directory containing WAN model weights",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root data directory — IntPhys (contains O1/O2/O3), InfLevel "
             "(contains continuity/gravity/solidity), or for --dataset phyworld "
             "the path to the parabola_eval.hdf5 file.",
    )
    parser.add_argument(
        "--dataset", type=str, default="intphys",
        choices=["intphys", "inflevel", "phyworld"],
        help="Which dataset format to load (default: intphys)",
    )
    parser.add_argument(
        "--meta-csv", type=Path, default=None,
        help="CSV file produced by create_intphys_csv.py "
             "(ignored when --dataset phyworld).",
    )
    parser.add_argument(
        "--phyworld-split", type=str, default="00000",
        help="Which split to load from the PhyWorld HDF5 — "
             "'00000' (1000 id), '00001' (56 ood), or 'all'.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Root output directory (default: outputs/)",
    )
    parser.add_argument(
        "--reverse_sample", action="store_true",
        help="Run DDIM inversion instead of forward generation",
    )
    parser.add_argument(
        "--encode-only", action="store_true",
        help="VAE-only baseline: encode videos to latents and save in probe-compatible format (no inversion).",
    )
    parser.add_argument(
        "--diagnose-inversion", action="store_true",
        help="Run forward→invert→forward on fixed-seed noise (skips VAE). Reports round-trip drift.",
    )
    parser.add_argument(
        "--reconstruct", action="store_true",
        help=(
            "After inversion, regenerate from the inverted noise and save "
            "original.mp4 + reconstructed.mp4 + MSE/PSNR in meta.json. "
            "Use with --reverse_sample. Best results with --guidance-scale 1.0."
        ),
    )
    parser.add_argument(
        "--i2v", action="store_true",
        help="Use the image-to-video pipeline (default: text-to-video)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Inference device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Diffusion steps (default: 50)",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=5.0,
        help="CFG scale (default: 5.0; use 1.0 for exact inversion)",
    )
    parser.add_argument(
        "--integrator", type=str, default="euler", choices=list(INTEGRATORS),
        help=(
            "Inversion integrator: euler makes one model call per step, heun "
            "makes two and averages them (default: euler). Text-to-video only."
        ),
    )
    parser.add_argument(
        "--negative-prompt", type=str, default="",
        help="Negative prompt applied to all samples",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Video height in pixels (default: 480)",
    )
    parser.add_argument(
        "--width", type=int, default=832,
        help="Video width in pixels (default: 832)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=81,
        help="Number of frames per video (default: 81)",
    )
    parser.add_argument(
        "--fps", type=int, default=16,
        help="FPS for written mp4s (default: 16)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Dataloader batch size (default: 1)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Dataloader worker processes (default: 0)",
    )
    parser.add_argument(
        "--temporal-pool", action="store_true",
        help=(
            "Pool spatial tokens per temporal frame in captured block outputs, "
            "reducing each tensor from (B, T*S, D) to (B, T, D). "
            "Preserves temporal structure while cutting storage ~74x."
        ),
    )
    parser.add_argument(
        "--capture-steps", nargs="*", type=int, default=None, metavar="STEP",
        help=(
            "Inversion step indices to capture block outputs for (0-indexed). "
            "Default: all steps. Example: --capture-steps 0 5 10 15 20 25 30 35 40 45"
        ),
    )
    parser.add_argument(
        "--capture-blocks", nargs="*", type=int, default=None, metavar="BLOCK",
        help=(
            "Transformer block indices to capture (0-indexed, 30 blocks total). "
            "Default: all blocks. Example: --capture-blocks 0 5 10 15 20 25"
        ),
    )
    parser.add_argument(
        "--save-every", type=int, default=None, metavar="N",
        help=(
            "Convenience shorthand: capture every N-th step (0, N, 2N, ...). "
            "Ignored if --capture-steps is explicitly specified."
        ),
    )
    parser.add_argument(
        "--capture-fractions", nargs="*", type=float, default=None, metavar="F",
        help=(
            "Capture at these fractions of the sigma schedule (0.0 = clean, 1.0 = max noise). "
            "Selects the step whose sigma is closest to each target. "
            "Ignored if --capture-steps is explicitly specified. "
            "Example: --capture-fractions 0.0 0.25 0.5 0.75 1.0"
        ),
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_arguments(argv)
    setup_logging(args.log_level)
    
    dtype = args.dtype
    if _DEBUG_FP32_TRANSFORMER:
        dtype = "float32"
        logger.warning("DEBUG: forcing fp32 transformer (_DEBUG_FP32_TRANSFORMER)")

    pipeline = (
        load_i2v_pipeline(args.ckpt_dir, args.device, dtype)
        if args.i2v
        else load_t2v_pipeline(args.ckpt_dir, args.device, dtype)
    )
    logger.info(f"pipeline.config.expand_timesteps = {pipeline.config.expand_timesteps}")
    logger.info(f"pipeline.config.boundary_ratio    = {pipeline.config.boundary_ratio}")
    logger.info(f"pipeline.transformer_2            = {getattr(pipeline, 'transformer_2', None) is not None}")
    logger.info(f"pipeline.scheduler type           = {type(pipeline.scheduler).__name__}")
    if _DEBUG_FORCE_EULER_SCHEDULER and not isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler):
        logger.warning(
            f"DEBUG: replacing {type(pipeline.scheduler).__name__} with FlowMatchEulerDiscreteScheduler "
            "(so inversion's Euler matches generation's integrator). "
            "Reconstruction quality will be visibly worse — turn off for normal runs."
        )
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
        logger.info(f"pipeline.scheduler type           = {type(pipeline.scheduler).__name__} (after swap)")

    # max_frames controls how many video frames are loaded per scene:
    #   None  : all frames (inversion / VAE-encode need every frame)
    #   1     : first frame only (I2V generation needs only the conditioning image)
    #   0     : no frames (T2V generation needs only prompts)
    if args.reverse_sample or args.encode_only:
        max_frames = None   # all frames
    elif args.i2v:
        max_frames = 1      # conditioning image only
    else:
        max_frames = 0      # prompts only

    dataloader = build_dataloader(
        data_dir=args.data_dir,
        meta_csv=str(args.meta_csv) if args.meta_csv is not None else "",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=max_frames,
        dataset=args.dataset,
        phyworld_split=args.phyworld_split,
    )

    run_dir = args.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.diagnose_inversion:
        run_inversion_diagnostic(
            pipeline=pipeline,
            num_inference_steps=args.steps,
            height=args.height, width=args.width, num_frames=args.num_frames,
            integrator=args.integrator,
        )
        return

    if args.encode_only:
        run_encoding(
            pipeline=pipeline,
            dataloader=dataloader,
            output_dir=run_dir,
            height=args.height,
            width=args.width,
        )
    elif args.reverse_sample:
        # Accumulate capture steps from all sources
        capture_steps = set()
        if args.capture_steps is not None:
            capture_steps.update(args.capture_steps)
        if args.capture_fractions is not None:
            capture_steps.update(resolve_capture_steps_from_fractions(
                pipeline.scheduler, args.steps, args.capture_fractions,
                device=pipeline._execution_device,
            ))
        if args.save_every is not None:
            capture_steps.update(range(0, args.steps, args.save_every))
        if not capture_steps:
            capture_steps = None  # none specified → capture all

        capture_blocks = (
            set(args.capture_blocks) if args.capture_blocks is not None else None
        )

        run_inversion(
            pipeline=pipeline,
            dataloader=dataloader,
            output_dir=run_dir,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            reconstruct=args.reconstruct,
            fps=args.fps,
            temporal_pool=args.temporal_pool,
            capture_steps=capture_steps,
            capture_blocks=capture_blocks,
            integrator=args.integrator,
        )
    else:
        run_generation(
            pipeline=pipeline,
            dataloader=dataloader,
            output_dir=run_dir,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
        )



if __name__ == "__main__":
    main()
