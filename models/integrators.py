"""
Euler and Heun integrators shared by the WAN, CogVideoX and LTX inversion loops.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch


INTEGRATORS = ("euler", "heun")


def integrate_step(
    latents: torch.Tensor,
    step_idx: int,
    num_steps: int,
    predict: Callable[[torch.Tensor, int], torch.Tensor],
    step: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    integrator: str = "euler",
    capture: Optional[ContextManager] = None,
    average: Optional[Callable[..., torch.Tensor]] = None,
) -> torch.Tensor:
    """One inversion step built from ``predict(latents, idx)`` and ``step(latents, prediction, idx)``. Heun predicts
    again at the Euler result with the next timestep and averages; ``capture`` wraps the first prediction only."""
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}; expected one of {INTEGRATORS}."
        )

    with capture if capture is not None else nullcontext():
        prediction = predict(latents, step_idx)

    if integrator == "heun" and step_idx + 1 < num_steps:
        latents_pred = step(latents, prediction, step_idx)
        prediction_next = predict(latents_pred, step_idx + 1)
        if average is None:
            prediction = 0.5 * (prediction + prediction_next)
        else:
            prediction = average(
                prediction, prediction_next, latents, latents_pred, step_idx
            )

    return step(latents, prediction, step_idx)
