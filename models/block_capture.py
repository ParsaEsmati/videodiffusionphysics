"""
Forward hooks that capture the output of every transformer block in a single
forward pass, used for the baseline encoders. The inversion loops use
``capture_block_outputs`` in ``wan_inversion.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch


@contextmanager
def capture_forward_hooks(
    blocks,
    block_indices: Optional[Iterable[int]] = None,
):
    """Context manager that hooks the given blocks and yields a dict ``{block_idx: output}``,
    filled as each block runs (float32, on CPU)."""
    captured: Dict[int, torch.Tensor] = {}
    selected = None if block_indices is None else set(block_indices)

    def make_hook(i: int):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[i] = hs.detach().float().cpu()
        return hook

    handles = []
    for i, block in enumerate(blocks):
        if selected is None or i in selected:
            handles.append(block.register_forward_hook(make_hook(i)))

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def find_transformer_blocks(model: torch.nn.Module):
    """Locate the list of transformer blocks in a Hugging Face encoder by trying the common
    attribute paths. Returns ``(blocks, path)``."""
    candidates = [
        "encoder.layer",
        "encoder.layers",
        "encoder.blocks",
        "blocks",
        "transformer_blocks",          # CogVideoX, FLUX
        "vision_model.encoder.layers",
        "vjepa2.encoder.layer",
        "videomae.encoder.layer",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            return obj, path
    raise RuntimeError(
        f"Could not find transformer blocks. Tried: {candidates}.\n"
        f"Inspect `print(model)` and pass the correct path via --blocks-path."
    )


def get_blocks_by_path(model: torch.nn.Module, path: str):
    """Walk a dotted attribute path to return the target ModuleList."""
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj
