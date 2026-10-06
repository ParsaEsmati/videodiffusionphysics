#!/usr/bin/env python3
"""
Block intervention scored with probe surprise, for WAN, CogVideoX and
LTX-Video, all text-to-video as inverted by ``inference.py``,
``cogvideox_inference.py`` and ``ltx_inference.py``.

For each plausible scene of an inversion run, the video is regenerated from the
scene's recovered noise (the baseline), then once per transformer block while
noise is added to that block's output at every denoising step:

    h  ->  h + alpha * std(h) * eps,    eps ~ N(0, I)

with ``std(h)`` the per-token standard deviation over features. Every video is
inverted again and scored with a probe trained by ``probe_pairwise.py`` at the
same step. The surprise of a video is logit(implausible) - logit(plausible),
averaged over the probe's blocks, and the result for block ``b`` is its shift
from the baseline: ``surprise(V_b) - surprise(V_base)``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from diffusers.utils import export_to_video

from cogvideox_inference import load_cogvideox_pipeline
from inference import load_t2v_pipeline
from ltx_inference import load_ltx_pipeline
from models.cogvideox_inversion import invert_cogvideox_t2v, prepare_video_for_inversion_cogvideox
from models.ltx_inversion import invert_ltx_t2v, prepare_video_for_inversion_ltx
from models.wan_inversion import invert_t2v, prepare_video_for_inversion
from probe_pairwise import PairwiseProbeModule


logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================


# Per model: the pipeline loader, the ``mode`` its inversion writes to meta.json, the prompt its inversion
# uses in place of an empty one, and the fps of the saved videos.
MODELS = {
    "wan":       dict(load=load_t2v_pipeline,       mode="T2V",           empty_prompt="",                     fps=16),
    "cogvideox": dict(load=load_cogvideox_pipeline, mode="T2V_cogvideox", empty_prompt="a video of an object", fps=8),
    "ltx":       dict(load=load_ltx_pipeline,       mode="T2V_ltx",       empty_prompt="a video",              fps=24),
}


def transformer_blocks(pipeline) -> torch.nn.ModuleList:
    """WAN keeps its blocks in ``.blocks``, CogVideoX and LTX in ``.transformer_blocks``."""
    transformer = pipeline.transformer
    return transformer.blocks if hasattr(transformer, "blocks") else transformer.transformer_blocks


def encode_prompt(pipeline, model: str, prompt: str, negative_prompt: Optional[str]) -> dict:
    """Prompt embeddings, as keyword arguments of the model's inversion function."""
    do_cfg = negative_prompt is not None
    out = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_cfg,
        device=pipeline._execution_device,
    )
    if model == "ltx":
        embeds, mask, neg_embeds, neg_mask = out
        return dict(
            prompt_embeds=embeds, prompt_attention_mask=mask,
            negative_prompt_embeds=neg_embeds if do_cfg else None,
            negative_prompt_attention_mask=neg_mask if do_cfg else None,
        )
    embeds, neg_embeds = out
    return dict(prompt_embeds=embeds, negative_prompt_embeds=neg_embeds if do_cfg else None)


def invert(pipeline, model: str, frames, meta: dict, prompt_kwargs: dict, step: int, step_callback) -> None:
    """Invert ``frames`` the way the run was inverted, handing the block outputs at ``step`` to
    ``step_callback``."""
    h, w = meta["height"], meta["width"]
    kwargs = dict(
        pipeline=pipeline,
        num_inference_steps=meta["num_inference_steps"],
        guidance_scale=meta["guidance_scale"],
        capture_hidden_states=True,
        capture_steps={step},
        save_trajectory=False,
        step_callback=step_callback,
        integrator=meta.get("integrator", "euler"),
        **prompt_kwargs,
    )
    if model == "wan":
        invert_t2v(video_latents=prepare_video_for_inversion(pipeline, frames, h, w), **kwargs)
    elif model == "cogvideox":
        invert_cogvideox_t2v(video_latents=prepare_video_for_inversion_cogvideox(pipeline, frames, h, w), **kwargs)
    else:
        invert_ltx_t2v(
            video_latents=prepare_video_for_inversion_ltx(pipeline, frames, h, w),
            height=h, width=w, num_frames=meta["num_frames"], frame_rate=meta["frame_rate"],
            **kwargs,
        )


# ============================================================================
# Intervention
# ============================================================================


@contextmanager
def noise_intervention(block: torch.nn.Module, alpha: float):
    """Add ``alpha * std(h) * eps`` to the output ``h`` of ``block`` on every forward pass."""

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out    # CogVideoX blocks also return the text states
        std = hs.std(dim=-1, keepdim=True).clamp(min=1e-6)
        hs = hs + alpha * std * torch.randn_like(hs)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ============================================================================
# Probe surprise
# ============================================================================


def load_probe(ckpt_path: Path, device) -> PairwiseProbeModule:
    """Load a ``probe_pairwise.py`` checkpoint for inference."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    probe = PairwiseProbeModule(**ckpt["hyper_parameters"])
    probe.load_state_dict(ckpt["state_dict"])
    return probe.to(device).eval()


class _ProbeStepReached(Exception):
    """Ends the inversion once the probe step has been captured."""


def probe_surprise(pipeline, model: str, probe, frames, meta: dict, prompt_kwargs: dict, step: int) -> float:
    """Invert ``frames`` up to ``step`` and return logit(implausible) - logit(plausible), averaged over blocks."""
    features = {}

    def on_step(_step_idx: int, step_data: dict) -> None:
        # Every block's output averaged over tokens: the (num_blocks, D) input the probe was trained on.
        features["x"] = torch.stack([step_data[k][0].mean(dim=0) for k in sorted(step_data)])
        raise _ProbeStepReached

    # The inversion loops print or draw a progress bar at every step.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            invert(pipeline, model, frames, meta, prompt_kwargs, step, on_step)
        except _ProbeStepReached:
            pass

    logits = probe(features["x"].unsqueeze(0).to(probe.device))[0]    # (num_blocks, 2); class 1 = plausible
    return float((logits[:, 0] - logits[:, 1]).mean())


# ============================================================================
# One scene
# ============================================================================


def plausible_scenes(run_dir: Path) -> list[Path]:
    """Scene folders of an inversion run with a recovered noise and label 1, in sorted order."""
    scenes = []
    for scene_dir in sorted(run_dir.iterdir()):
        meta_path = scene_dir / "meta.json"
        if (scene_dir / "noise_latent.pt").is_file() and meta_path.is_file():
            if int(json.loads(meta_path.read_text())["label"]) == 1:
                scenes.append(scene_dir)
    return scenes


@torch.no_grad()
def run_scene(pipeline, probe, scene_dir: Path, out_dir: Path, blocks: list[int], args) -> dict:
    """Score the baseline video and the video with each block intervened on."""
    meta = json.loads((scene_dir / "meta.json").read_text())
    device = pipeline._execution_device
    noise = torch.load(scene_dir / "noise_latent.pt", map_location=device, weights_only=True)
    noise = noise.to(pipeline.transformer.dtype)

    # Same prompt as the inversion, so the baseline reproduces the inverted video.
    prompt = meta["prompt"] if meta["prompt"].strip() else MODELS[args.model]["empty_prompt"]
    negative_prompt = args.negative_prompt if meta["guidance_scale"] > 1.0 else None
    prompt_kwargs = encode_prompt(pipeline, args.model, prompt, negative_prompt)
    gen_kwargs = dict(
        prompt=prompt, negative_prompt=negative_prompt,
        num_inference_steps=meta["num_inference_steps"], guidance_scale=meta["guidance_scale"],
        height=meta["height"], width=meta["width"], num_frames=meta["num_frames"],
    )
    if args.model == "ltx":
        gen_kwargs["frame_rate"] = meta["frame_rate"]

    def generate_and_score(name: str) -> float:
        frames = pipeline(latents=noise.clone(), **gen_kwargs).frames[0]
        if args.save_videos:
            export_to_video(frames, str(out_dir / f"{name}.mp4"), fps=MODELS[args.model]["fps"])
        return probe_surprise(pipeline, args.model, probe, frames, meta, prompt_kwargs, args.step)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)    # makes the intervention noise reproducible per scene
    baseline = generate_and_score("baseline")
    logger.info(f"    baseline  surprise={baseline:+.4f}")

    block_surprise = {}
    for b in blocks:
        with noise_intervention(transformer_blocks(pipeline)[b], args.alpha):
            block_surprise[str(b)] = generate_and_score(f"block_{b:02d}")
        logger.info(f"    block {b:02d}  delta={block_surprise[str(b)] - baseline:+.4f}")

    return {
        "scene": scene_dir.name,
        "model": args.model,
        "alpha": args.alpha,
        "step": args.step,
        "probe_ckpt": str(args.probe_ckpt.resolve()),
        "seed": args.seed,
        "baseline": baseline,
        "blocks": block_surprise,
    }


# ============================================================================
# Summary
# ============================================================================


def write_summary(output_dir: Path, top_k: int = 8) -> None:
    """Gather the finished scenes into ``surprise_per_scene.csv`` and average the shift of every block over them
    into ``surprise_per_block.csv``."""
    rows = []
    for path in sorted(output_dir.glob("*/surprise.json")):
        result = json.loads(path.read_text())
        for b, surprise in result["blocks"].items():
            rows.append({
                "scene": result["scene"],
                "block": int(b),
                "delta_surprise": surprise - result["baseline"],
            })
    if not rows:
        return

    per_scene = pd.DataFrame(rows)
    per_scene.to_csv(output_dir / "surprise_per_scene.csv", index=False)
    per_block = (
        per_scene.groupby("block")["delta_surprise"]
        .agg(mean_delta_surprise="mean", std_delta_surprise="std", num_scenes="count")
        .reset_index()
    )
    per_block.to_csv(output_dir / "surprise_per_block.csv", index=False)

    top = per_block.nlargest(top_k, "mean_delta_surprise")
    logger.info(f"Wrote {output_dir / 'surprise_per_block.csv'} over {per_scene['scene'].nunique()} scene(s)")
    logger.info(f"Top {top_k} blocks: {', '.join(f'{b:02d}' for b in top['block'])}")


# ============================================================================
# CLI
# ============================================================================


def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS),
                        help="Model the run was inverted with")
    parser.add_argument("--ckpt-dir", type=Path, required=True,
                        help="Checkpoint directory of that model")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Timestamped inversion folder written in step 2")
    parser.add_argument("--probe-ckpt", type=Path, required=True,
                        help="Probe checkpoint written by probe_pairwise.py")
    parser.add_argument("--step", type=int, required=True,
                        help="Inversion step the probe was trained on")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Noise strength relative to the per-token std (default: 0.5)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results folder (default: <run-dir>/intervention_step<NNNN>); "
                             "finished scenes in it are skipped")
    parser.add_argument("--blocks", nargs="*", type=int, default=None, metavar="BLOCK",
                        help="Blocks to intervene on (default: all)")
    parser.add_argument("--num-scenes", type=int, default=None,
                        help="Only run the first N plausible scenes (default: all)")
    parser.add_argument("--shard", type=int, default=0,
                        help="Run every --num-shards-th scene starting at this one, to split a run over jobs")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the intervention noise, reset for every scene")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt, used only if the inversion run had guidance > 1")
    parser.add_argument("--save-videos", action="store_true",
                        help="Also save baseline.mp4 and block_NN.mp4 for each scene")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")
    args = parse_arguments(argv)
    output_dir = args.output_dir or args.run_dir / f"intervention_step{args.step:04d}"

    scenes = plausible_scenes(args.run_dir)[args.shard::args.num_shards][:args.num_scenes]
    if not scenes:
        raise SystemExit(f"No plausible scenes with noise_latent.pt found in {args.run_dir}")
    meta = json.loads((scenes[0] / "meta.json").read_text())
    if meta.get("mode") != MODELS[args.model]["mode"]:
        raise SystemExit(f"{args.run_dir} was inverted in mode {meta.get('mode')!r}, "
                         f"but --model {args.model} expects {MODELS[args.model]['mode']!r}")
    if args.step >= meta["num_inference_steps"]:
        raise SystemExit(f"--step {args.step} is past the {meta['num_inference_steps']} inversion steps of the run")

    pipeline = MODELS[args.model]["load"](args.ckpt_dir, args.device, args.dtype)
    pipeline.set_progress_bar_config(disable=True)
    probe = load_probe(args.probe_ckpt, pipeline._execution_device)
    num_blocks = len(transformer_blocks(pipeline))
    if probe.num_blocks != num_blocks:
        raise SystemExit(f"Probe has {probe.num_blocks} blocks but {args.model} has {num_blocks}")
    blocks = args.blocks if args.blocks else list(range(num_blocks))

    logger.info(f"{args.model}: {len(scenes)} plausible scene(s)  blocks={blocks}  "
                f"alpha={args.alpha}  probe step={args.step}  -> {output_dir}")
    for i, scene_dir in enumerate(scenes, start=1):
        out_dir = output_dir / scene_dir.name
        done = out_dir / "surprise.json"
        if done.is_file():
            previous = json.loads(done.read_text())
            if (previous["alpha"], previous["step"], previous["probe_ckpt"]) != (
                args.alpha, args.step, str(args.probe_ckpt.resolve())
            ):
                raise SystemExit(f"{done} was run with another --alpha, --step or --probe-ckpt; "
                                 f"pass a new --output-dir")
            logger.info(f"  [{i}/{len(scenes)}] {scene_dir.name}: already done, skipping")
            continue
        logger.info(f"  [{i}/{len(scenes)}] {scene_dir.name}")
        result = run_scene(pipeline, probe, scene_dir, out_dir, blocks, args)
        with open(done, "w") as f:
            json.dump(result, f, indent=2)

    write_summary(output_dir)


if __name__ == "__main__":
    main()
