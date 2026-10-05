"""
DDIM inversion for CogVideoX: takes a clean video latent to noise by running
the deterministic DDIM update in the reverse direction, with optional capture
of the transformer block outputs.
"""

from __future__ import annotations

import torch
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from models.cogvideox_pipeline import CogVideoXPipeline
from models.integrators import integrate_step
from models.wan_inversion import capture_block_outputs   # model-agnostic hook helper


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class CogInversionOutput:
    """Result of running inversion on a video latent.

    Attributes:
        noise_latent:  Recovered noise after the last inversion step,
                       ``(B, T', z_dim, H', W')``.
        trajectory:    List of latents saved after each inversion step
                       (only populated when ``save_trajectory=True``).
        block_outputs: ``{step_idx: {block_idx: tensor}}`` populated when
                       ``capture_hidden_states=True``. Each tensor is
                       ``(B, seq, hidden_dim)`` on CPU.
        timesteps:     Reversed timestep sequence used (ascending).
    """
    noise_latent: torch.Tensor
    trajectory: List[torch.Tensor] = field(default_factory=list)
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    timesteps: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


# ---------------------------------------------------------------------------
# VAE encode / decode helpers (mirror pipeline's prepare_latents + decode_latents)
# ---------------------------------------------------------------------------


def encode_video_to_latents_cogvideox(
    pipeline: CogVideoXPipeline,
    video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    seed: int = 0,
) -> torch.Tensor:
    """VAE-encode a preprocessed video into ``(B, T', z_dim, H', W')`` latents. Samples the posterior with
    a fixed seed (not its mode), so the latent variance matches what the model saw in training."""
    device = device or pipeline._execution_device
    dtype = dtype or pipeline.vae.dtype

    video = video.to(device=device, dtype=dtype)
    generator = torch.Generator(device=video.device).manual_seed(seed)
    with torch.no_grad():
        latents = pipeline.vae.encode(video).latent_dist.sample(generator)   # (B, C, T, H, W)

    latents = latents * pipeline.vae_scaling_factor_image            # match pipeline convention
    latents = latents.permute(0, 2, 1, 3, 4).contiguous()            # -> (B, T, C, H, W)
    return latents.to(dtype=torch.float32)


def prepare_video_for_inversion_cogvideox(
    pipeline: CogVideoXPipeline,
    video_frames,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Preprocess and VAE-encode a video for CogVideoX inversion."""
    device = device or pipeline._execution_device

    if isinstance(video_frames, torch.Tensor) and video_frames.ndim == 4:
        video_frames = video_frames.permute(1, 0, 2, 3)              # (C, T, H, W) -> (T, C, H, W)

    video = pipeline.video_processor.preprocess_video(
        video_frames, height=height, width=width
    )
    return encode_video_to_latents_cogvideox(pipeline, video, device=device)


def decode_latents_cogvideox_to_pil(pipeline: CogVideoXPipeline, latents: torch.Tensor):
    """Decode latents to a list of PIL frames."""
    video = pipeline.decode_latents(latents)
    return pipeline.video_processor.postprocess_video(video=video, output_type="pil")[0]


# ---------------------------------------------------------------------------
# DDIM inversion schedule
# ---------------------------------------------------------------------------


def _build_ddim_inversion_schedule(
    scheduler,
    num_inference_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(inv_timesteps, inv_alphas)``: the N timesteps in ascending noise order, and the
    N + 1 cumulative alphas of the states, starting at the clean latent."""
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps.to(device)                       # (N,), descending t
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    fwd_alphas = alphas_cumprod[timesteps.long()]                    # (N,) alpha at each forward timestep

    inv_timesteps = timesteps.flip(0)                                # ascending t  -> more noise
    inv_alphas = fwd_alphas.flip(0)                                  # descending alpha -> more noise

    final_alpha = getattr(scheduler, "final_alpha_cumprod", torch.tensor(1.0))
    if not isinstance(final_alpha, torch.Tensor):
        final_alpha = torch.tensor(final_alpha)
    final_alpha = final_alpha.to(device=device, dtype=inv_alphas.dtype)

    inv_alphas_padded = torch.cat([final_alpha.unsqueeze(0), inv_alphas])   # (N+1,)
    return inv_timesteps, inv_alphas_padded


def _ddim_invert_step(
    x_curr: torch.Tensor,
    model_output: torch.Tensor,
    alpha_curr: torch.Tensor,
    alpha_next: torch.Tensor,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """One DDIM step from the noise level ``alpha_curr`` to the noisier ``alpha_next``."""
    sqrt_a_curr = alpha_curr.sqrt()
    sqrt_1m_a_curr = (1 - alpha_curr).sqrt()
    sqrt_a_next = alpha_next.sqrt()
    sqrt_1m_a_next = (1 - alpha_next).sqrt()

    if prediction_type == "epsilon":
        eps = model_output
        pred_x0 = (x_curr - sqrt_1m_a_curr * eps) / sqrt_a_curr.clamp_min(1e-8)
    elif prediction_type == "v_prediction":
        v = model_output
        pred_x0 = sqrt_a_curr * x_curr - sqrt_1m_a_curr * v
        eps = sqrt_1m_a_curr * x_curr + sqrt_a_curr * v
    elif prediction_type == "sample":
        pred_x0 = model_output
        eps = (x_curr - sqrt_a_curr * pred_x0) / sqrt_1m_a_curr.clamp_min(1e-8)
    else:
        raise ValueError(
            f"Unsupported prediction_type {prediction_type!r} "
            "(expected 'epsilon', 'v_prediction', or 'sample')."
        )

    return sqrt_a_next * pred_x0 + sqrt_1m_a_next * eps


def _model_output_to_eps(
    x: torch.Tensor,
    model_output: torch.Tensor,
    alpha: torch.Tensor,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """Noise estimate implied by ``model_output`` for a latent ``x`` at level ``alpha``."""
    sqrt_a = alpha.sqrt()
    sqrt_1m_a = (1 - alpha).sqrt()

    if prediction_type == "epsilon":
        return model_output
    if prediction_type == "v_prediction":
        return sqrt_1m_a * x + sqrt_a * model_output
    if prediction_type == "sample":
        return (x - sqrt_a * model_output) / sqrt_1m_a.clamp_min(1e-8)
    raise ValueError(
        f"Unsupported prediction_type {prediction_type!r} "
        "(expected 'epsilon', 'v_prediction', or 'sample')."
    )


def _eps_to_model_output(
    x: torch.Tensor,
    eps: torch.Tensor,
    alpha: torch.Tensor,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """Inverse of ``_model_output_to_eps``: the model output that implies ``eps``."""
    sqrt_a = alpha.sqrt()
    sqrt_1m_a = (1 - alpha).sqrt()

    if prediction_type == "epsilon":
        return eps
    if prediction_type == "v_prediction":
        return (eps - sqrt_1m_a * x) / sqrt_a.clamp_min(1e-8)
    if prediction_type == "sample":
        return (x - sqrt_1m_a * eps) / sqrt_a.clamp_min(1e-8)
    raise ValueError(
        f"Unsupported prediction_type {prediction_type!r} "
        "(expected 'epsilon', 'v_prediction', or 'sample')."
    )


def _average_ddim_outputs(
    model_output: torch.Tensor,
    x_curr: torch.Tensor,
    alpha_curr: torch.Tensor,
    model_output_next: torch.Tensor,
    x_next: torch.Tensor,
    alpha_next: torch.Tensor,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """Heun average of two model outputs taken at different noise levels: both are converted to
    noise estimates, averaged, and converted back to a model output for ``x_curr``."""
    eps = 0.5 * (
        _model_output_to_eps(x_curr, model_output, alpha_curr, prediction_type)
        + _model_output_to_eps(x_next, model_output_next, alpha_next, prediction_type)
    )
    return _eps_to_model_output(x_curr, eps, alpha_curr, prediction_type)


# ---------------------------------------------------------------------------
# Core inversion (T2V)
# ---------------------------------------------------------------------------


def invert_cogvideox_t2v(
    pipeline: CogVideoXPipeline,
    video_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    capture_hidden_states: bool = False,
    capture_steps: Optional[set] = None,
    capture_blocks: Optional[set] = None,
    save_trajectory: bool = True,
    attention_kwargs: Optional[dict] = None,
    step_callback: Optional[callable] = None,
    integrator: str = "euler",
) -> CogInversionOutput:
    """Invert CogVideoX latents ``(B, T', z_dim, H', W')`` from clean to noise. Block outputs at
    ``capture_steps`` are handed to ``step_callback(step_idx, {block_idx: tensor})``."""
    device = pipeline._execution_device
    transformer = pipeline.transformer
    transformer_dtype = transformer.dtype
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
    prediction_type = getattr(pipeline.scheduler.config, "prediction_type", "v_prediction")

    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if do_cfg:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        # Pipeline order is [neg, pos] on dim 0; we replicate that exactly.
        cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    else:
        cfg_prompt_embeds = prompt_embeds

    inv_timesteps, inv_alphas = _build_ddim_inversion_schedule(
        pipeline.scheduler, num_inference_steps, device
    )

    latents = video_latents.clone().to(device=device, dtype=torch.float32)
    trajectory = [latents.clone().cpu()] if save_trajectory else []
    block_outputs: Dict[int, Dict[int, torch.Tensor]] = {}

    # Mirror pipeline step 7: rotary positional embeddings.
    image_rotary_emb = (
        pipeline._prepare_rotary_positional_embeddings(
            height=video_latents.shape[3] * pipeline.vae_scale_factor_spatial,
            width=video_latents.shape[4] * pipeline.vae_scale_factor_spatial,
            num_frames=video_latents.shape[1],
            device=device,
        )
        if getattr(transformer.config, "use_rotary_positional_embeddings", False)
        else None
    )

    def predict(latents: torch.Tensor, idx: int) -> torch.Tensor:
        """Guided model output for ``latents`` at ``inv_timesteps[idx]``, as float32."""
        t = inv_timesteps[idx]

        # Pipeline mirrors: scale_model_input + (optional) cat for CFG.
        if do_cfg:
            latent_model_input = torch.cat([latents] * 2, dim=0)
        else:
            latent_model_input = latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(transformer_dtype)

        timestep = t.expand(latent_model_input.shape[0])
        with torch.no_grad():
            with transformer.cache_context("cond_uncond" if do_cfg else "cond"):
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=cfg_prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
        noise_pred = noise_pred.float()                              # match pipeline cast

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        return noise_pred

    def step(latents: torch.Tensor, model_output: torch.Tensor, idx: int) -> torch.Tensor:
        """DDIM update from ``inv_alphas[idx]`` to ``inv_alphas[idx + 1]``."""
        return _ddim_invert_step(
            latents, model_output, inv_alphas[idx], inv_alphas[idx + 1],
            prediction_type=prediction_type,
        )

    def average(
        model_output: torch.Tensor,
        model_output_next: torch.Tensor,
        latents: torch.Tensor,
        latents_pred: torch.Tensor,
        idx: int,
    ) -> torch.Tensor:
        """Heun average of the outputs at ``inv_alphas[idx]`` and ``inv_alphas[idx + 1]``."""
        return _average_ddim_outputs(
            model_output, latents, inv_alphas[idx],
            model_output_next, latents_pred, inv_alphas[idx + 1],
            prediction_type=prediction_type,
        )

    for step_idx, t in enumerate(tqdm(inv_timesteps, desc="cogvideox-invert", leave=False)):
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
            average=average,
        )

        if save_trajectory:
            trajectory.append(latents.clone().cpu())

        if should_capture and step_idx in block_outputs:
            if step_callback is not None:
                step_callback(step_idx, block_outputs[step_idx])
            del block_outputs[step_idx]

    return CogInversionOutput(
        noise_latent=latents.clone(),
        trajectory=trajectory,
        block_outputs=block_outputs,
        timesteps=inv_timesteps,
    )
