"""Dataset loading and activation caching utilities."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

HOOK_POINTS = [
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_post",
    "blocks.1.hook_resid_post",
]


def load_model(device: str = "cuda") -> HookedTransformer:
    """Load gelu-2l and move it to the specified device."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project requires a CUDA GPU.")
    model = HookedTransformer.from_pretrained("gelu-2l", device=device)
    return model


def extract_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_points: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Run a forward pass and return activations at the specified hook points."""
    if hook_points is None:
        hook_points = HOOK_POINTS

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_points,
        )

    return {hp: cache[hp] for hp in hook_points}
