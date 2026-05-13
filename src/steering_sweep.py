"""Steering strength sweep: characterize coherence/steering tradeoff."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import mlflow
import torch
from sae_lens import SAE

from src.data import load_model
from src.mlflow_utils import init_mlflow, start_run
from src.steering_demo import (
    MAX_NEW_TOKENS,
    PROMPTS,
    STEERING_FEATURES,
    TEMPERATURE,
    TOP_K,
    generate_with_hook,
    load_config,
    make_steering_hook,
    set_seed,
)

SWEEP_STRENGTHS = [1, 2, 5, 10, 15, 20]


def generate_baselines(
    model: Any,
    prompts: list[str],
    hook_point: str,
    tokenizer: Any,
    seed: int,
) -> dict[str, str]:
    """Generate baseline text for each prompt (no hook). Run once."""
    baselines: dict[str, str] = {}
    for prompt_text in prompts:
        prompt_tokens = model.to_tokens(prompt_text)
        tokens = generate_with_hook(
            model, prompt_tokens, hook_point, None,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed,
        )
        baselines[prompt_text] = tokenizer.decode(tokens[0].tolist())
    return baselines


def run_sweep_for_feature(
    model: Any,
    sae: SAE,
    hook_point: str,
    feature_idx: int,
    prompts: list[str],
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    """Run all (strength, prompt) pairs for one feature."""
    results: list[dict[str, Any]] = []
    for strength in SWEEP_STRENGTHS:
        hook_fn = make_steering_hook(sae, feature_idx, "amplify", strength)
        for prompt_text in prompts:
            prompt_tokens = model.to_tokens(prompt_text)
            tokens = generate_with_hook(
                model, prompt_tokens, hook_point, hook_fn,
                MAX_NEW_TOKENS, TEMPERATURE, TOP_K, seed,
            )
            text = tokenizer.decode(tokens[0].tolist())
            results.append({
                "strength": strength,
                "prompt": prompt_text,
                "text": text,
            })
    return results


def strength_to_color(strength: float, max_strength: float = 20.0) -> str:
    """Map strength to a CSS background color (green -> yellow -> red)."""
    t = min(strength / max_strength, 1.0)
    if t < 0.5:
        # green to yellow
        r = int(200 * (t * 2))
        g = 200
    else:
        # yellow to red
        r = 200
        g = int(200 * (1 - (t - 0.5) * 2))
    return f"rgb({r}, {g}, 180)"


def render_feature_section(
    feat_label: str,
    feat_desc: str,
    results: list[dict[str, Any]],
    baselines: dict[str, str],
    prompts: list[str],
) -> list[str]:
    """Render HTML section for one feature: baselines + strength table."""
    lines = [
        "<div class='feature-section'>",
        f"<h2>{html.escape(feat_label)}</h2>",
        f"<p class='desc'>{html.escape(feat_desc)}</p>",
    ]

    # Build lookup: (strength, prompt) -> text
    lookup: dict[tuple[float, str], str] = {}
    for r in results:
        lookup[(r["strength"], r["prompt"])] = r["text"]

    for prompt_text in prompts:
        lines.append("<div class='prompt-block'>")
        lines.append(f"<h3>Prompt: \"{html.escape(prompt_text)}\"</h3>")

        # Baseline
        lines.append("<div class='baseline-box'>")
        lines.append("<b>Baseline:</b><br>")
        lines.append(f"<span class='gen-text'>{html.escape(baselines[prompt_text])}</span>")
        lines.append("</div>")

        # Strength rows
        for strength in SWEEP_STRENGTHS:
            color = strength_to_color(strength)
            text = lookup.get((strength, prompt_text), "")
            lines.append(
                f"<div class='strength-row' style='background:{color};'>"
                f"<span class='str-label'>str={strength}</span>"
                f"<span class='gen-text'>{html.escape(text)}</span>"
                f"</div>"
            )

        lines.append("</div>")  # prompt-block

    lines.append("</div>")  # feature-section
    return lines


def generate_sweep_html(
    hook_point: str,
    baselines: dict[str, str],
    all_results: dict[str, list[dict[str, Any]]],
    feature_descriptions: dict[str, str],
) -> str:
    """Assemble the full HTML report for the steering sweep."""
    lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>Steering Sweep: {html.escape(hook_point)}</title>",
        "<style>",
        "body { font-family: sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }",
        "h3 { color: #444; margin-top: 20px; }",
        ".desc { color: #777; font-style: italic; }",
        ".info { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0; }",
        ".feature-section { margin: 20px 0; }",
        ".prompt-block { margin: 15px 0; }",
        ".baseline-box { background: #e8f5e9; border: 1px solid #a5d6a7; padding: 10px; "
        "border-radius: 6px; margin: 8px 0; }",
        ".strength-row { padding: 8px 12px; margin: 4px 0; border-radius: 4px; "
        "display: flex; gap: 12px; align-items: flex-start; }",
        ".str-label { font-weight: bold; min-width: 60px; flex-shrink: 0; }",
        ".gen-text { font-family: monospace; font-size: 0.85em; white-space: pre-wrap; "
        "word-wrap: break-word; }",
        "</style>",
        "</head><body>",
        f"<h1>Steering Strength Sweep: {html.escape(hook_point)}</h1>",
        "<div class='info'>",
        f"<p><b>Hook point:</b> {html.escape(hook_point)}</p>",
        f"<p><b>Strengths tested:</b> {SWEEP_STRENGTHS}</p>",
        f"<p><b>Features:</b> {len(all_results)}</p>",
        f"<p><b>Prompts:</b> {len(baselines)}</p>",
        f"<p><b>Max new tokens:</b> {MAX_NEW_TOKENS}, "
        f"Temperature: {TEMPERATURE}, Top-k: {TOP_K}</p>",
        "</div>",
    ]

    for feat_label, results in all_results.items():
        desc = feature_descriptions.get(feat_label, "")
        section = render_feature_section(
            feat_label, desc, results, baselines, PROMPTS,
        )
        lines.extend(section)

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Steering strength sweep")
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

    print(f"Steering sweep for: {hook_point}")
    print(f"Strengths: {SWEEP_STRENGTHS}")

    # Load model and SAE once
    model = load_model(device="cuda")
    sae = SAE.load_from_disk(str(sae_path), device="cuda")
    tokenizer = model.tokenizer

    # Generate baselines once
    print("Generating baselines...")
    baselines = generate_baselines(model, PROMPTS, hook_point, tokenizer, args.seed)

    # Sweep each feature
    all_results: dict[str, list[dict[str, Any]]] = {}
    feature_descriptions: dict[str, str] = {}

    for feat_idx, feat_label, feat_desc in STEERING_FEATURES:
        print(f"\n--- Sweeping feature {feat_idx}: {feat_label} ---")
        feature_descriptions[feat_label] = f"Feature {feat_idx}: {feat_desc}"
        results = run_sweep_for_feature(
            model, sae, hook_point, feat_idx, PROMPTS, tokenizer, args.seed,
        )
        all_results[feat_label] = results

    # Generate HTML report
    print("\nGenerating sweep report...")
    report_html = generate_sweep_html(
        hook_point, baselines, all_results, feature_descriptions,
    )

    report_dir = Path("outputs") / "steering_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"sweep_{hook_slug}.html"
    report_path.write_text(report_html)
    print(f"Report saved to: {report_path}")

    # Log to MLflow
    init_mlflow()
    with start_run(
        run_name=f"steering_sweep_{hook_slug}",
        hook_point=hook_point,
        run_purpose="steering_sweep",
    ):
        mlflow.log_params({
            "hook_point": hook_point,
            "sweep_strengths": str(SWEEP_STRENGTHS),
            "n_features": len(STEERING_FEATURES),
            "n_prompts": len(PROMPTS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "seed": args.seed,
            "feature_indices": str([f[0] for f in STEERING_FEATURES]),
        })
        mlflow.log_artifact(str(report_path), "steering_reports")

    print("MLflow run logged.")

    # Cleanup
    del model, sae
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
