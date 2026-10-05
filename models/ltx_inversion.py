"""
Flow-matching inversion for LTX-Video: takes packed clean video latents to
noise on the pipeline's own sigma schedule, with optional capture of the
transformer block outputs.
"""

from __future__ import annotations

import torch
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from models.integrators import integrate_step
from models.ltx_pipeline import LTXPipeline, calculate_shift, retrieve_timesteps
from models.wan_inversion import capture_block_outputs    # model-agnostic forward-hook helper


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class LTXInversionOutput:
    """Result of running inversion on a video latent.

    Attributes:
        noise_latent:  Recovered noise at sigma_max in PACKED format
                       (B, num_patches, packed_C). Pass directly back to
                       ``LTXPipeline(latents=...)`` for reconstruction.
        trajectory:    List of packed latents saved after each inversion step
                       (only populated when ``save_trajectory=True``).
        block_outputs: ``{step_idx: {block_idx: tensor}}`` populated when
                       ``capture_hidden_states=True``. Each tensor is
                       ``(B, packed_seq, hidden_dim)`` on CPU.
        timesteps:     Reversed timestep sequence used (ascending).
    """
    noise_latent: torch.Tensor
    trajectory: List[torch.Tensor] = field(default_factory=list)
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    timesteps: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


# ---------------------------------------------------------------------------
# VAE encode / pack helpers (mirror the pipeline exactly)
# ---------------------------------------------------------------------------


def encode_video_to_latents_ltx(
    pipeline: LTXPipeline,
    video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """VAE-encode a preprocessed video, normalize the latents and pack them to ``(B, num_patches, packed_C)``."""
    device = device or pipeline._execution_device
    dtype = dtype or pipeline.vae.dtype

    video = video.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = pipeline.vae.encode(video).latent_dist.mode()    # (B, C, T, H, W)

    # LTX: normalize using per-channel mean/std AND a scalar scaling factor.
    latents = LTXPipeline._normalize_latents(
        latents,
        pipeline.vae.latents_mean,
        pipeline.vae.latents_std,
        pipeline.vae.config.scaling_factor,
    )

    # Pack to (B, num_patches, packed_C) — the transformer's input shape.
    latents = LTXPipeline._pack_latents(
        latents,
        pipeline.transformer_spatial_patch_size,
        pipeline.transformer_temporal_patch_size,
    )
    return latents.to(dtype=torch.float32)


def prepare_video_for_inversion_ltx(
    pipeline: LTXPipeline,
    video_frames,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess, VAE-encode and pack a video (PIL list, array or ``(C, T, H, W)`` tensor) for inversion."""
    device = device or pipeline._execution_device
    if isinstance(video_frames, torch.Tensor) and video_frames.ndim == 4:
        video_frames = video_frames.permute(1, 0, 2, 3)    # (C, T, H, W) -> (T, C, H, W)
    video = pipeline.video_processor.preprocess_video(
        video_frames, height=height, width=width
    )
    return encode_video_to_latents_ltx(pipeline, video, device=device)


def decode_latents_ltx_to_pil(
    pipeline: LTXPipeline,
    packed_latents: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    decode_timestep: float = 0.05,
    decode_noise_scale: Optional[float] = 0.025,
):
    """Unpack and decode packed latents to a list of PIL frames."""
    latents = LTXPipeline._unpack_latents(
        packed_latents,
        latent_num_frames, latent_height, latent_width,
        pipeline.transformer_spatial_patch_size,
        pipeline.transformer_temporal_patch_size,
    )
    latents = LTXPipeline._denormalize_latents(
        latents, pipeline.vae.latents_mean, pipeline.vae.latents_std,
        pipeline.vae.config.scaling_factor,
    )
    latents = latents.to(pipeline.vae.dtype)

    timestep = None
    if pipeline.vae.config.timestep_conditioning:
        device = latents.device
        b = latents.shape[0]
        timestep = torch.tensor([decode_timestep] * b, device=device, dtype=latents.dtype)
        if decode_noise_scale is not None:
            noise = torch.randn_like(latents)
            scale = torch.tensor([decode_noise_scale] * b,
                                 device=device, dtype=latents.dtype)[:, None, None, None, None]
            latents = (1 - scale) * latents + scale * noise

    with torch.no_grad():
        video = pipeline.vae.decode(latents, timestep, return_dict=False)[0]
    return pipeline.video_processor.postprocess_video(video, output_type="pil")[0]


# ---------------------------------------------------------------------------
# Inversion schedule
# ---------------------------------------------------------------------------


def _build_ltx_inversion_schedule(
    pipeline: LTXPipeline,
    num_inference_steps: int,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(inv_timesteps, inv_sigmas)`` in ascending noise order, from the same shifted schedule
    the pipeline uses; ``inv_sigmas`` has N + 1 entries and starts at 0."""
    video_sequence_length = latent_num_frames * latent_height * latent_width
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        video_sequence_length,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device, None,
        sigmas=sigmas, mu=mu,
    )
    fwd_timesteps = timesteps.to(device)                                      # (N,) descending
    fwd_sigmas = pipeline.scheduler.sigmas.to(device)                         # (N+1,) descending

    inv_timesteps = fwd_timesteps.flip(0)                                     # ascending
    inv_sigmas = fwd_sigmas.flip(0)                                           # ascending  -> noise
    return inv_timesteps, inv_sigmas


# ---------------------------------------------------------------------------
# Core inversion (T2V)
# ---------------------------------------------------------------------------


def invert_ltx_t2v(
    pipeline: LTXPipeline,
    video_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    capture_hidden_states: bool = False,
    capture_steps: Optional[set] = None,
    capture_blocks: Optional[set] = None,
    save_trajectory: bool = True,
    attention_kwargs: Optional[dict] = None,
    step_callback: Optional[callable] = None,
    height: int = 480,
    width: int = 704,
    num_frames: int = 161,
    frame_rate: int = 25,
    integrator: str = "euler",
) -> LTXInversionOutput:
    """Invert packed LTX latents ``(B, num_patches, packed_C)`` from clean to noise. Block outputs at
    ``capture_steps`` are handed to ``step_callback(step_idx, {block_idx: tensor})``."""
    device = pipeline._execution_device
    transformer = pipeline.transformer
    transformer_dtype = transformer.dtype
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(device)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)
        cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        cfg_attn_mask = torch.cat(
            [negative_prompt_attention_mask, prompt_attention_mask], dim=0
        )
    else:
        cfg_prompt_embeds = prompt_embeds
        cfg_attn_mask = prompt_attention_mask

    # Latent grid + RoPE interpolation scale (mirrors pipeline lines 717-749).
    latent_num_frames = (num_frames - 1) // pipeline.vae_temporal_compression_ratio + 1
    latent_height = height // pipeline.vae_spatial_compression_ratio
    latent_width = width // pipeline.vae_spatial_compression_ratio
    rope_interpolation_scale = (
        pipeline.vae_temporal_compression_ratio / frame_rate,
        pipeline.vae_spatial_compression_ratio,
        pipeline.vae_spatial_compression_ratio,
    )

    inv_timesteps, inv_sigmas = _build_ltx_inversion_schedule(
        pipeline, num_inference_steps,
        latent_num_frames, latent_height, latent_width,
        device,
    )

    latents = video_latents.clone().to(device=device, dtype=torch.float32)
    trajectory = [latents.clone().cpu()] if save_trajectory else []
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = {}

    def predict(latents: torch.Tensor, idx: int) -> torch.Tensor:
        """Guided velocity for ``latents`` at ``inv_timesteps[idx]``, as float32."""
        if do_cfg:
            latent_model_input = torch.cat([latents] * 2, dim=0)
        else:
            latent_model_input = latents
        latent_model_input = latent_model_input.to(transformer_dtype)
        timestep = inv_timesteps[idx].expand(latent_model_input.shape[0])
        with torch.no_grad():
            with transformer.cache_context("cond_uncond" if do_cfg else "cond"):
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=cfg_prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=cfg_attn_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
        noise_pred = noise_pred.float()

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        return noise_pred

    def step(latents: torch.Tensor, velocity: torch.Tensor, idx: int) -> torch.Tensor:
        """Flow-matching update from ``inv_sigmas[idx]`` to ``inv_sigmas[idx + 1]``."""
        dt = inv_sigmas[idx + 1] - inv_sigmas[idx]            # > 0 for inversion
        return latents + dt * velocity

    for step_idx, t in enumerate(tqdm(inv_timesteps, desc="ltx-invert", leave=False)):
        should_capture = capture_hidden_states and (
            capture_steps is None or step_idx in capture_steps
        )
        ctx = (
            capture_block_outputs(
                transformer, block_outputs, step_idx,
                temporal_pool=False, T=1, S=1,
                capture_blocks=capture_blocks,
            )
            if should_capture
            else nullcontext()
        )

        latents = integrate_step(
            latents, step_idx, len(inv_timesteps),
            predict=predict, step=step, integrator=integrator, capture=ctx,
        )

        if save_trajectory:
            trajectory.append(latents.clone().cpu())

        if should_capture and step_idx in block_outputs:
            if step_callback is not None:
                step_callback(step_idx, block_outputs[step_idx])
            del block_outputs[step_idx]

    return LTXInversionOutput(
        noise_latent=latents.clone(),
        trajectory=trajectory,
        block_outputs=block_outputs,
        timesteps=inv_timesteps,
    )
