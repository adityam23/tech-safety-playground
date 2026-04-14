"""Fast smoke tests: load model, extract activations, verify shapes, log to MLflow."""

from __future__ import annotations

import torch
import mlflow

from src.data import HOOK_POINTS, extract_activations, load_model
from src.mlflow_utils import init_mlflow, start_run


def test_cuda_available() -> None:
    """CUDA must be available for this project."""
    assert torch.cuda.is_available(), "CUDA is not available"


def test_load_model_and_extract_activations() -> None:
    """Load gelu-2l, run a single batch, check activation shapes, log to MLflow."""
    model = load_model(device="cuda")
    d_model = model.cfg.d_model

    # Create a small random batch: (batch=2, seq_len=16)
    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len), device="cuda")

    activations = extract_activations(model, tokens)

    # Verify we got all three hook points with correct shapes
    for hp in HOOK_POINTS:
        assert hp in activations, f"Missing hook point: {hp}"
        shape = activations[hp].shape
        assert shape == (batch_size, seq_len, d_model), (
            f"{hp}: expected ({batch_size}, {seq_len}, {d_model}), got {shape}"
        )

    # Log shapes and VRAM to MLflow
    init_mlflow()
    with start_run(
        run_name="smoke_test",
        hook_point="all",
        run_purpose="smoke_test",
    ):
        for hp in HOOK_POINTS:
            shape = activations[hp].shape
            mlflow.log_params(
                {
                    f"{hp}_shape_batch": shape[0],
                    f"{hp}_shape_seq": shape[1],
                    f"{hp}_shape_d_model": shape[2],
                }
            )
        mlflow.log_params(
            {
                "model_name": "gelu-2l",
                "d_model": d_model,
                "d_vocab": model.cfg.d_vocab,
                "n_layers": model.cfg.n_layers,
            }
        )
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mlflow.log_metric("peak_vram_mb", peak_vram_mb)

    # Cleanup
    del activations, model
    torch.cuda.empty_cache()
