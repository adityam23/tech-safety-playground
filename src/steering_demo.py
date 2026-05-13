"""Steering demo: amplify/suppress SAE features and generate text."""

from __future__ import annotations

import argparse
import html
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from sae_lens import SAE

from src.data import load_model
from src.mlflow_utils import init_mlflow, start_run

# Features identified from Phase 4 analysis (blocks.0.hook_resid_post)
# Each entry: (feature_index, label, description)
STEERING_FEATURES = [
    (26, "frequency_adverbs", "Activates on adverbs like 'generally', 'usually', 'commonly', 'often'"),
    (103, "need_to_pattern", "Activates on 'need to' infinitive constructions"),
    (420, "money_financial", "Activates on money/financial terms like 'money', 'dollars'"),
    (131, "tech_brands", "Activates on tech products/brands like Apple Watch, Galaxy, Xiaomi"),
    (384, "reference_regard", "Activates on reference/regard words like 'reference', 'respect', 'regard'"),
]

PROMPTS = [
    "The most important thing about technology is",
    "In order to improve our society, we should",
    "The weather today is expected to be",
    "When cooking a meal, you should always",
    "Scientists have recently discovered that",
]

MAX_NEW_TOKENS = 60
AMPLIFY_STRENGTH = 20.0  # magnitude added along feature direction
TEMPERATURE = 0.8
TOP_K = 50


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_steering_hook(
    sae: SAE,
    feature_idx: int,
    mode: str,
    steering_strength: float,
) -> Any:
    """Create a TransformerLens hook that steers via an SAE feature direction.

    For "amplify": adds steering_strength * decoder_direction to the residual stream.
    For "ablate": projects out the feature direction from the residual stream.

    Args:
        sae: The trained SAE.
        feature_idx: Which feature to steer.
        mode: "amplify" or "ablate".
        steering_strength: Magnitude to add along the feature direction (amplify mode).

    Returns:
        A hook function compatible with model.add_hook().
    """
    # Get the decoder direction for this feature (the "concept vector")
    steering_vec = sae.W_dec.data[feature_idx].clone()  # (d_model,)
    steering_vec = steering_vec / steering_vec.norm()  # normalize

    def hook_fn(activations: torch.Tensor, hook: Any) -> torch.Tensor:
        # activations shape: (batch, seq, d_model)
        if mode == "amplify":
            # Add the feature direction scaled by steering_strength
            activations = activations + steering_strength * steering_vec
        elif mode == "ablate":
            # Project out the feature direction from all positions
            # proj = (act · d) * d, where d is the unit direction
            dots = (activations * steering_vec).sum(dim=-1, keepdim=True)
            activations = activations - dots * steering_vec
        return activations

    return hook_fn


def generate_with_hook(
    model: Any,
    prompt_tokens: torch.Tensor,
    hook_point: str,
    hook_fn: Any | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> torch.Tensor:
    """Generate tokens with an optional hook applied at every forward pass."""
    set_seed(seed)
    tokens = prompt_tokens.clone()

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            if hook_fn is not None:
                model.reset_hooks()
                model.add_hook(hook_point, hook_fn)

            logits = model(tokens)

            if hook_fn is not None:
                model.reset_hooks()

        # Sample from the last position
        next_logits = logits[0, -1, :] / temperature
        # Top-k filtering
        topk_vals, topk_idx = torch.topk(next_logits, top_k)
        probs = torch.zeros_like(next_logits).fill_(float("-inf"))
        probs.scatter_(0, topk_idx, topk_vals)
        probs = torch.softmax(probs, dim=0)
        next_token = torch.multinomial(probs, 1).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def run_steering_for_feature(
    model: Any,
    sae: SAE,
    hook_point: str,
    feature_idx: int,
    feature_label: str,
    prompts: list[str],
    tokenizer: Any,
    seed: int,
) -> list[dict[str, str]]:
    """Run baseline, amplified, and ablated generation for one feature."""
    results = []

    for prompt_text in prompts:
        prompt_tokens = model.to_tokens(prompt_text)  # (1, seq_len)

        # Baseline (no hook)
        baseline_tokens = generate_with_hook(
            model, prompt_tokens, hook_point, None,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed,
        )
        baseline_text = tokenizer.decode(baseline_tokens[0].tolist())

        # Amplified
        amp_hook = make_steering_hook(sae, feature_idx, "amplify", AMPLIFY_STRENGTH)
        amp_tokens = generate_with_hook(
            model, prompt_tokens, hook_point, amp_hook,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed,
        )
        amp_text = tokenizer.decode(amp_tokens[0].tolist())

        # Ablated
        abl_hook = make_steering_hook(sae, feature_idx, "ablate", 0.0)
        abl_tokens = generate_with_hook(
            model, prompt_tokens, hook_point, abl_hook,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed,
        )
        abl_text = tokenizer.decode(abl_tokens[0].tolist())

        results.append(
            {
                "prompt": prompt_text,
                "baseline": baseline_text,
                "amplified": amp_text,
                "ablated": abl_text,
            }
        )

    return results


def generate_steering_html(
    hook_point: str,
    all_results: dict[str, list[dict[str, str]]],
    feature_descriptions: dict[str, str],
) -> str:
    """Generate an HTML report with side-by-side steering outputs."""
    lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>Steering Demo: {html.escape(hook_point)}</title>",
        "<style>",
        "body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }",
        "h3 { color: #666; }",
        ".feature-section { margin: 20px 0; }",
        ".prompt-group { margin: 15px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }",
        ".prompt { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; color: #222; }",
        ".outputs { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }",
        ".output-box { padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; word-wrap: break-word; }",
        ".baseline { background: #e8f5e9; border: 1px solid #a5d6a7; }",
        ".amplified { background: #fff3e0; border: 1px solid #ffcc80; }",
        ".ablated { background: #e3f2fd; border: 1px solid #90caf9; }",
        ".label { font-weight: bold; margin-bottom: 5px; font-family: sans-serif; font-size: 0.9em; }",
        ".desc { color: #777; font-style: italic; margin-bottom: 10px; }",
        ".info { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0; }",
        "</style>",
        "</head><body>",
        f"<h1>Steering Demo: {html.escape(hook_point)}</h1>",
        "<div class='info'>",
        f"<p><b>Hook point:</b> {html.escape(hook_point)}</p>",
        f"<p><b>Amplification multiplier:</b> {AMPLIFY_STRENGTH}x</p>",
        f"<p><b>Max new tokens:</b> {MAX_NEW_TOKENS}</p>",
        f"<p><b>Temperature:</b> {TEMPERATURE}</p>",
        f"<p><b>Features tested:</b> {len(all_results)}</p>",
        "</div>",
    ]

    for feat_label, results in all_results.items():
        desc = feature_descriptions.get(feat_label, "")
        lines.append("<div class='feature-section'>")
        lines.append(f"<h2>Feature: {html.escape(feat_label)}</h2>")
        lines.append(f"<p class='desc'>{html.escape(desc)}</p>")

        for res in results:
            lines.append("<div class='prompt-group'>")
            lines.append(f"<div class='prompt'>Prompt: \"{html.escape(res['prompt'])}\"</div>")
            lines.append("<div class='outputs'>")

            lines.append("<div class='output-box baseline'>")
            lines.append("<div class='label'>Baseline</div>")
            lines.append(html.escape(res["baseline"]))
            lines.append("</div>")

            lines.append("<div class='output-box amplified'>")
            lines.append(f"<div class='label'>Amplified ({AMPLIFY_STRENGTH}x)</div>")
            lines.append(html.escape(res["amplified"]))
            lines.append("</div>")

            lines.append("<div class='output-box ablated'>")
            lines.append("<div class='label'>Ablated</div>")
            lines.append(html.escape(res["ablated"]))
            lines.append("</div>")

            lines.append("</div>")  # outputs
            lines.append("</div>")  # prompt-group

        lines.append("</div>")  # feature-section

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAE steering demo")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project requires a CUDA GPU.")

    set_seed(args.seed)

    cfg = load_config(args.config)
    hook_point = cfg["hook_point"]
    hook_slug = hook_point.replace(".", "_")
    sae_path = Path("outputs") / f"sae_{hook_slug}"

    print(f"Steering demo for: {hook_point}")
    print(f"SAE path: {sae_path}")

    # Load model and SAE
    model = load_model(device="cuda")
    sae = SAE.load_from_disk(str(sae_path), device="cuda")
    tokenizer = model.tokenizer

    all_results: dict[str, list[dict[str, str]]] = {}
    feature_descriptions: dict[str, str] = {}

    for feat_idx, feat_label, feat_desc in STEERING_FEATURES:
        print(f"\n--- Feature {feat_idx}: {feat_label} ---")
        print(f"    {feat_desc}")
        feature_descriptions[feat_label] = f"Feature {feat_idx}: {feat_desc}"

        results = run_steering_for_feature(
            model, sae, hook_point,
            feat_idx, feat_label, PROMPTS,
            tokenizer, args.seed,
        )
        all_results[feat_label] = results

        # Print a quick preview
        for res in results[:1]:
            print(f"  Prompt: {res['prompt'][:50]}...")
            print(f"  Baseline: {res['baseline'][:80]}...")
            print(f"  Amplified: {res['amplified'][:80]}...")
            print(f"  Ablated: {res['ablated'][:80]}...")

    # Generate HTML report
    print("\nGenerating HTML report...")
    report_html = generate_steering_html(hook_point, all_results, feature_descriptions)

    report_dir = Path("outputs") / "steering_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"steering_{hook_slug}.html"
    report_path.write_text(report_html)
    print(f"Report saved to: {report_path}")

    # Log to MLflow
    init_mlflow()
    with start_run(
        run_name=f"steering_demo_{hook_slug}",
        hook_point=hook_point,
        run_purpose="steering_demo",
    ):
        mlflow.log_params(
            {
                "hook_point": hook_point,
                "n_features": len(STEERING_FEATURES),
                "n_prompts": len(PROMPTS),
                "amplify_multiplier": AMPLIFY_STRENGTH,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "seed": args.seed,
                "feature_indices": str([f[0] for f in STEERING_FEATURES]),
            }
        )
        mlflow.log_artifact(str(report_path), "steering_reports")

    print("MLflow run logged.")

    # Cleanup
    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
