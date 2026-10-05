"""
Flow-matching inversion for WAN: takes a clean video latent to noise by
running the sampler's update on the reversed sigma schedule, with optional
capture of the transformer block outputs.
"""

from __future__ import annotations

import torch
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from diffusers.utils.torch_utils import randn_tensor

from models.block_capture import find_transformer_blocks
from models.integrators import integrate_step


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class InversionOutput:
    """Result of running inversion on a video latent.

    Attributes:
        noise_latent:   Recovered noise at sigma_max (the final state after
                        all inversion steps). Shape: same as input latents.
        trajectory:     List of latent tensors saved after each inversion step.
                        trajectory[0] = x_clean (original encoded video)
                        trajectory[-1] = noise_latent
                        Length = num_inference_steps + 1
        block_outputs:  Nested dict: step_idx -> block_idx -> hidden state tensor
                        (B, seq_len, hidden_dim) float32 on CPU.
                        Populated when capture_hidden_states=True.
        timesteps:      The reversed timestep sequence used (ascending sigma).
    """
    noise_latent: torch.Tensor
    trajectory: List[torch.Tensor] = field(default_factory=list)
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    timesteps: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


# ---------------------------------------------------------------------------
# Block output capture via forward hooks
# ---------------------------------------------------------------------------

@contextmanager
def capture_block_outputs(
    transformer,
    outputs_out: Dict[int, Dict[int, torch.Tensor]],
    step: int,
    temporal_pool: bool = False,
    T: int = 21,
    S: int = 1560,
    capture_blocks: Optional[set] = None,
):
    """Context manager that hooks the transformer blocks and writes their outputs for one
    inversion step into ``outputs_out[step][block_idx]`` (float32, on CPU)."""
    step_outputs: Dict[int, torch.Tensor] = {}
    outputs_out[step] = step_outputs

    def make_hook(block_idx: int):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            if temporal_pool:
                B, _, D = hs.shape
                hs = hs.view(B, T, S, D).mean(dim=2)  # (B, T, D)
            step_outputs[block_idx] = hs.detach().float().cpu()
        return hook

    # Autodetect block list: WAN uses `.blocks`, CogVideoX uses `.transformer_blocks`, etc.
    if hasattr(transformer, "blocks"):
        blocks = transformer.blocks
    elif hasattr(transformer, "transformer_blocks"):
        blocks = transformer.transformer_blocks
    else:
        blocks, _ = find_transformer_blocks(transformer)

    handles = [
        block.register_forward_hook(make_hook(i))
        for i, block in enumerate(blocks)
        if capture_blocks is None or i in capture_blocks
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# VAE encode helpers
# ---------------------------------------------------------------------------

def encode_video_to_latents(
    pipeline,
    video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """VAE-encode a preprocessed video ``(B, C, T, H, W)`` and normalize the latents.
    Returns ``(B, z_dim, T', H', W')``."""
    device = device or pipeline._execution_device
    dtype = dtype or pipeline.vae.dtype

    video = video.to(device=device, dtype=dtype)

    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(pipeline.vae.config.latents_std)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(device, dtype)
    )

    with torch.no_grad():
        # Use mode (argmax) for deterministic encoding
        latents = pipeline.vae.encode(video).latent_dist.mode()

    latents = (latents - latents_mean) * latents_std


    # DEBUG: round-trip decode to verify encode/decode correctness
    # latents_unnorm = latents / latents_std + latents_mean
    # with torch.no_grad():
    #     decoded = pipeline.vae.decode(latents_unnorm, return_dict=False)[0]
    # from diffusers.utils import export_to_video
    # export_to_video(pipeline.video_processor.postprocess_video(video, output_type="pil")[0], "debug_original.mp4", fps=16)
    # export_to_video(pipeline.video_processor.postprocess_video(decoded, output_type="pil")[0], "debug_reconstructed.mp4", fps=16)
    # END DEBUG


    return latents.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Core inversion functions
# ---------------------------------------------------------------------------

def _build_inversion_schedule(
    scheduler,
    num_inference_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(inv_timesteps, inv_sigmas)`` in ascending noise order; ``inv_sigmas`` has
    N + 1 entries, from 0 (clean) to sigma_max."""
    scheduler.set_timesteps(num_inference_steps, device=device)

    # scheduler.sigmas: [sigma_max, ..., sigma_min, 0]  (length = N+1)
    # scheduler.timesteps: [t_max, ..., t_min]  (length = N)
    fwd_sigmas = scheduler.sigmas        # shape [N+1]
    fwd_timesteps = scheduler.timesteps  # shape [N]

    # Reverse: small-sigma first -> large-sigma last
    inv_sigmas = fwd_sigmas.flip(0)        # [0, sigma_min, ..., sigma_max]
    inv_timesteps = fwd_timesteps.flip(0)  # [t_min, ..., t_max]

    return inv_timesteps, inv_sigmas


def invert_t2v(
    pipeline,
    video_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    capture_hidden_states: bool = False,
    temporal_pool: bool = False,
    capture_steps: Optional[set] = None,
    capture_blocks: Optional[set] = None,
    save_trajectory: bool = True,
    attention_kwargs: Optional[dict] = None,
    step_callback: Optional[callable] = None,
    integrator: str = "euler",
) -> InversionOutput:
    """Invert WAN text-to-video latents ``(B, C, T', H', W')`` from clean to noise. Block outputs at
    ``capture_steps`` are handed to ``step_callback(step_idx, {block_idx: tensor})``."""
    device = pipeline._execution_device
    transformer = pipeline.transformer
    transformer_dtype = transformer.dtype
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    inv_timesteps, inv_sigmas = _build_inversion_schedule(
        pipeline.scheduler, num_inference_steps, device
    )

    latents = video_latents.clone().to(device=device, dtype=torch.float32)
    trajectory = [latents.clone().cpu()] if save_trajectory else []
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = {}

    # Token-space dims derived from latent shape; patch_size=(1,2,2) is model-fixed
    _, _, T_lat, H_lat, W_lat = video_latents.shape
    T_tok = T_lat
    S_tok = (H_lat // 2) * (W_lat // 2)

    def predict(latents: torch.Tensor, idx: int) -> torch.Tensor:
        """Guided velocity for ``latents`` at ``inv_timesteps[idx]``, as float32."""
        timestep = inv_timesteps[idx].expand(latents.shape[0])
        latent_input = latents.to(transformer_dtype)
        with torch.no_grad():
            with transformer.cache_context("cond"):
                noise_pred = transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if do_cfg:
                with transformer.cache_context("uncond"):
                    noise_uncond = transformer(
                        hidden_states=latent_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        return noise_pred.float()

    def step(latents: torch.Tensor, velocity: torch.Tensor, idx: int) -> torch.Tensor:
        """Flow-matching update from ``inv_sigmas[idx]`` to ``inv_sigmas[idx + 1]``."""
        dt = inv_sigmas[idx + 1] - inv_sigmas[idx]  # positive: adding noise
        return latents + dt * velocity

    for step_idx, t in enumerate(inv_timesteps):
        print(f"Inversion step {step_idx+1}/{len(inv_timesteps)}: t={t.item():.4f}, sigma={inv_sigmas[step_idx].item():.4f}")

        should_capture = capture_hidden_states and (
            capture_steps is None or step_idx in capture_steps
        )
        ctx = (
            capture_block_outputs(
                transformer, block_outputs, step_idx,
                temporal_pool=temporal_pool, T=T_tok, S=S_tok,
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

        # Flush captured step to callback and free memory
        if should_capture and step_idx in block_outputs:
            if step_callback is not None:
                step_callback(step_idx, block_outputs[step_idx])
            del block_outputs[step_idx]

    return InversionOutput(
        noise_latent=latents.clone(),
        trajectory=trajectory,
        block_outputs=block_outputs,
        timesteps=inv_timesteps,
    )


def manual_generate_t2v(
    pipeline,
    noise_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    integrator: str = "euler",
    attention_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Denoise with the same Euler or Heun update as ``invert_t2v``, in the opposite direction.
    Returns the clean latent without decoding it."""
    device = pipeline._execution_device
    transformer = pipeline.transformer
    transformer_dtype = transformer.dtype
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas = pipeline.scheduler.sigmas          # [sigma_max, ..., sigma_min, 0], length N+1
    timesteps = pipeline.scheduler.timesteps    # [t_max, ..., t_min],            length N

    latents = noise_latents.clone().to(device=device, dtype=torch.float32)

    for step_idx, t in enumerate(timesteps):
        sigma_curr = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma_curr            # negative for forward denoise

        timestep = t.expand(latents.shape[0])
        latent_input = latents.to(transformer_dtype)

        with torch.no_grad():
            with transformer.cache_context("cond"):
                noise_pred = transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            if do_cfg:
                with transformer.cache_context("uncond"):
                    noise_uncond = transformer(
                        hidden_states=latent_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        if integrator == "heun" and step_idx + 1 < len(timesteps):
            latents_pred = latents + dt * noise_pred.float()
            t_next = timesteps[step_idx + 1].expand(latents.shape[0])
            latent_pred_input = latents_pred.to(transformer_dtype)
            with torch.no_grad():
                with transformer.cache_context("cond"):
                    noise_pred_2 = transformer(
                        hidden_states=latent_pred_input,
                        timestep=t_next,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                if do_cfg:
                    with transformer.cache_context("uncond"):
                        noise_uncond_2 = transformer(
                            hidden_states=latent_pred_input,
                            timestep=t_next,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred_2 = noise_uncond_2 + guidance_scale * (noise_pred_2 - noise_uncond_2)
            noise_pred_avg = 0.5 * (noise_pred.float() + noise_pred_2.float())
            latents = latents + dt * noise_pred_avg
        else:
            latents = latents + dt * noise_pred.float()

    return latents


def decode_latents_to_video(pipeline, latents: torch.Tensor):
    """Un-normalize and VAE-decode a latent to a list of PIL frames, matching
    the pipeline's own post-processing path."""
    latents = latents.to(pipeline.vae.dtype)
    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
        1, pipeline.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        video = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.video_processor.postprocess_video(video, output_type="pil")[0]


def invert_i2v(
    pipeline,
    video_latents: torch.Tensor,
    condition: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    capture_hidden_states: bool = False,
    temporal_pool: bool = False,
    capture_steps: Optional[set] = None,
    capture_blocks: Optional[set] = None,
    save_trajectory: bool = True,
    attention_kwargs: Optional[dict] = None,
    step_callback: Optional[callable] = None,
) -> InversionOutput:
    """Invert WAN image-to-video latents from clean to noise, given the image-to-video condition
    tensor. Same capture options as ``invert_t2v``."""
    device = pipeline._execution_device
    transformer = pipeline.transformer
    transformer_dtype = transformer.dtype
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    inv_timesteps, inv_sigmas = _build_inversion_schedule(
        pipeline.scheduler, num_inference_steps, device
    )

    latents = video_latents.clone().to(device=device, dtype=torch.float32)
    condition = condition.to(device=device, dtype=transformer_dtype)

    trajectory = [latents.clone().cpu()] if save_trajectory else []
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = {}

    # Token-space dims derived from latent shape; patch_size=(1,2,2) is model-fixed
    _, _, T_lat, H_lat, W_lat = video_latents.shape
    T_tok = T_lat
    S_tok = (H_lat // 2) * (W_lat // 2)

    for step_idx, t in enumerate(inv_timesteps):
        sigma_curr = inv_sigmas[step_idx]
        sigma_next = inv_sigmas[step_idx + 1]
        dt = sigma_next - sigma_curr  # positive

        timestep = t.expand(latents.shape[0])

        # I2V concatenates condition along channel dim; cast to transformer_dtype for the forward pass
        latent_model_input = torch.cat([latents.to(transformer_dtype), condition], dim=1)

        should_capture = capture_hidden_states and (
            capture_steps is None or step_idx in capture_steps
        )
        ctx = (
            capture_block_outputs(
                transformer, block_outputs, step_idx,
                temporal_pool=temporal_pool, T=T_tok, S=S_tok,
                capture_blocks=capture_blocks,
            )
            if should_capture
            else nullcontext()
        )

        with torch.no_grad():
            with ctx:
                with transformer.cache_context("cond"):
                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_cfg:
                    with transformer.cache_context("uncond"):
                        noise_uncond = transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        latents = latents + dt * noise_pred.float()

        if save_trajectory:
            trajectory.append(latents.clone().cpu())

        # Flush captured step to callback and free memory
        if should_capture and step_idx in block_outputs:
            if step_callback is not None:
                step_callback(step_idx, block_outputs[step_idx])
            del block_outputs[step_idx]

    return InversionOutput(
        noise_latent=latents.clone(),
        trajectory=trajectory,
        block_outputs=block_outputs,
        timesteps=inv_timesteps,
    )


# ---------------------------------------------------------------------------
# High-level convenience wrappers
# ---------------------------------------------------------------------------

def prepare_video_for_inversion(
    pipeline,
    video_frames,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess and VAE-encode a video (PIL list, array or ``(C, T, H, W)`` tensor).
    Returns ``(1, z_dim, T', H', W')`` latents."""
    device = device or pipeline._execution_device

    if isinstance(video_frames, torch.Tensor) and video_frames.ndim == 4:
        video_frames = video_frames.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)

    video = pipeline.video_processor.preprocess_video(video_frames, height=height, width=width)
    return encode_video_to_latents(pipeline, video, device=device)


def prepare_i2v_condition(
    pipeline,
    first_frame,
    height: int,
    width: int,
    num_frames: int,
    device: Optional[torch.device] = None,
    last_frame=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build the image-to-video condition (mask and first-frame latent) from a PIL image.
    Returns ``(condition, image_embeds)``."""
    device = device or pipeline._execution_device
    dtype = torch.float32

    # Pre-process image(s)
    image = pipeline.video_processor.preprocess(first_frame, height=height, width=width)
    image = image.to(device=device, dtype=dtype)

    if last_frame is not None:
        last_image = pipeline.video_processor.preprocess(last_frame, height=height, width=width)
        last_image = last_image.to(device=device, dtype=dtype)
    else:
        last_image = None

    # Use prepare_latents but with a dummy noise tensor (we only need the condition)
    dummy_latents = torch.zeros(
        1,
        pipeline.vae.config.z_dim,
        (num_frames - 1) // pipeline.vae_scale_factor_temporal + 1,
        height // pipeline.vae_scale_factor_spatial,
        width // pipeline.vae_scale_factor_spatial,
        dtype=dtype,
        device=device,
    )

    result = pipeline.prepare_latents(
        image=image,
        batch_size=1,
        num_channels_latents=pipeline.vae.config.z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=dtype,
        device=device,
        generator=None,
        latents=dummy_latents,
        last_image=last_image,
    )

    if pipeline.config.expand_timesteps:
        _, condition, _ = result
    else:
        _, condition = result

    # CLIP image embeds (wan2.1 i2v only)
    image_embeds = None
    if (
        pipeline.transformer is not None
        and pipeline.transformer.config.image_dim is not None
        and pipeline.image_encoder is not None
    ):
        with torch.no_grad():
            if last_frame is None:
                image_embeds = pipeline.encode_image(first_frame, device)
            else:
                image_embeds = pipeline.encode_image([first_frame, last_frame], device)
        image_embeds = image_embeds.to(pipeline.transformer.dtype)

    return condition, image_embeds
