"""Model loading, activation extraction, and activation injection for Activation Oracles."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


PLACEHOLDER_TOKEN = " ?"


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model


def apply_lora(model: nn.Module, config: dict) -> nn.Module:
    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def get_injection_layer(model: nn.Module) -> int:
    return 1


def get_activation_layer(model: nn.Module, depth_frac: float = 0.5) -> int:
    return int(depth_frac * model.config.num_hidden_layers)


def extract_activation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    text: str,
    depth_frac: float = 0.5,
) -> torch.Tensor:
    """Extract a single activation vector from the target model.

    Returns activation at the *last* token position, shape (d_model,).
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    target_layer = get_activation_layer(model, depth_frac)

    captured: torch.Tensor | None = None

    def hook_fn(_module, _input, output):
        nonlocal captured
        if isinstance(output, tuple):
            captured = output[0].detach().clone()
        else:
            captured = output.detach().clone()

    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        handle.remove()

    if captured is None:
        raise RuntimeError("Hook did not fire for activation extraction")
    return captured[0, -1, :]  # (d_model,)


def make_steering_hook(
    activation: torch.Tensor,
    placeholder_positions: list[int] | None = None,
):
    """Create an injection hook for activation steering at layer 1.

    Applies the additive steering equation at each placeholder position:
      h' = h + ||h|| * v / ||v||

    If placeholder_positions is None, injects at ALL positions (for smoke testing).
    """
    v = activation.detach().clone()  # (d_model,)
    v_norm = v.norm(p=2)

    def hook_fn(_module, _input, output):
        if isinstance(output, tuple):
            h = output[0].clone()  # (batch, seq, d_model)
        else:
            h = output.clone()
        h_norms = h.norm(p=2, dim=-1, keepdim=True)  # (batch, seq, 1)

        if placeholder_positions is not None:
            for pos in placeholder_positions:
                if pos < h.shape[1]:
                    steer = h_norms[0, pos, :] * v / (v_norm + 1e-8)
                    h[0, pos, :] = h[0, pos, :] + steer
        else:
            steer = h_norms * v.unsqueeze(0).unsqueeze(0) / (v_norm + 1e-8)
            h = h + steer

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    return hook_fn


def find_placeholder_positions(
    input_ids: torch.Tensor,
    placeholder_id: int,
) -> list[int]:
    """Find positions of placeholder tokens in the tokenized input."""
    pos = (input_ids[0] == placeholder_id).nonzero(as_tuple=True)[0]
    return pos.tolist()


def generate_with_steering(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt_ids: torch.Tensor,
    activation: torch.Tensor,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    inject_all: bool = False,
) -> torch.Tensor:
    """Generate text with activation steering injected at layer 1."""
    injection_layer = get_injection_layer(model)
    placeholder_id = tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]

    placeholder_positions = find_placeholder_positions(prompt_ids, placeholder_id)
    if inject_all:
        placeholder_positions = None  # inject at ALL positions

    steer_hook = make_steering_hook(activation, placeholder_positions)

    handle = model.model.layers[injection_layer].register_forward_hook(steer_hook)
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    return output_ids


def format_oracle_prompt(
    activation_layer: int,
    n_placeholders: int,
    question: str,
) -> str:
    """Format the oracle prompt with layer info and placeholder tokens.

    Format: Layer: {layer}\n ? ? ?\n{question}
    """
    placeholders = PLACEHOLDER_TOKEN * n_placeholders
    return f"Layer: {activation_layer}\n{placeholders}\n{question}"
