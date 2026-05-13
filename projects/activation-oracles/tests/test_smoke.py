"""Smoke test: load Qwen2.5-0.5B, extract activation, verify steering changes output."""

from __future__ import annotations

import torch

from src.model import (
    _resolve_layers,
    extract_activation,
    generate_with_steering,
    get_activation_layer,
    load_base_model,
    load_tokenizer,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def test_cuda_available() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"


def test_load_model_and_extract_activation() -> None:
    """Load Qwen2.5-0.5B, extract an activation, verify shape."""
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_base_model(MODEL_NAME)

    layers = _resolve_layers(model)
    n_layers = len(layers)
    assert n_layers == 24, f"Expected 24 layers, got {n_layers}"

    act_layer = get_activation_layer(model, 0.5)
    assert act_layer == 12

    prompt = "The movie was fantastic and I really enjoyed it."
    activation = extract_activation(model, tokenizer, prompt, depth_frac=0.5)

    assert activation.ndim == 1
    assert activation.shape[0] == 896
    assert activation.dtype == torch.bfloat16

    del model, tokenizer
    torch.cuda.empty_cache()


def test_steering_hook_fires() -> None:
    """Verify the injection hook fires and modifies hidden states at layer 1."""
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_base_model(MODEL_NAME)

    text = "The movie was fantastic."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Run baseline forward pass, capture layer 1 output
    baseline_h = None

    def capture_baseline(_module, _input, output):
        nonlocal baseline_h
        baseline_h = output[0].detach().clone()

    h1 = _resolve_layers(model)[1].register_forward_hook(capture_baseline)
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        h1.remove()

    # Now run with a strong random steering vector
    activation = torch.randn(896, dtype=torch.bfloat16, device=model.device)
    steered_h = None

    def capture_steered(_module, _input, output):
        nonlocal steered_h
        steered_h = output[0].detach().clone()

    from src.model import make_steering_hook
    steer_hook_fn = make_steering_hook(activation, placeholder_positions=None)
    h_s = _resolve_layers(model)[1].register_forward_hook(steer_hook_fn)
    h_c = _resolve_layers(model)[1].register_forward_hook(capture_steered)
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        h_s.remove()
        h_c.remove()

    assert baseline_h is not None
    assert steered_h is not None
    diff = (steered_h - baseline_h).abs().max().item()
    print(f"\nMax steering delta at layer 1: {diff:.4f}")
    assert diff > 0.01, f"Steering had no effect (delta={diff})"

    del model, tokenizer
    torch.cuda.empty_cache()


def test_steering_changes_generation() -> None:
    """Verify that strong activation injection changes generated output."""
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_base_model(MODEL_NAME)

    question = "What is 1 + 1?"
    prompt = question
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Baseline generation
    with torch.inference_mode():
        baseline_ids = model.generate(
            prompt_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    baseline_text = tokenizer.decode(
        baseline_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True
    )

    # Steered generation with random activation at ALL positions
    activation = torch.randn(896, dtype=torch.bfloat16, device=model.device)
    steered_ids = generate_with_steering(
        model, tokenizer, prompt_ids, activation,
        max_new_tokens=10, inject_all=True,
    )
    steered_text = tokenizer.decode(
        steered_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True
    )

    print(f"\nBaseline: {baseline_text!r}")
    print(f"Steered:  {steered_text!r}")

    # With random activation injected everywhere, output SHOULD differ
    assert baseline_text != steered_text, (
        "Steering had no effect on generation"
    )

    del model, tokenizer
    torch.cuda.empty_cache()
