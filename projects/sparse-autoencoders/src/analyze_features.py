"""Feature analysis: dead features, top activations, feature categorization."""

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
from datasets import load_dataset
from safetensors.torch import load_file as st_load_file
from sae_lens import SAE

from src.data import load_model
from src.mlflow_utils import init_mlflow, start_run

N_SAMPLE_FEATURES = 50
N_TOP_EXAMPLES = 20
N_HELD_OUT_BATCHES = 200
BATCH_SIZE_PROMPTS = 16
CONTEXT_LENGTH = 128


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


def get_dead_feature_stats(
    sparsity_path: Path,
) -> tuple[torch.Tensor, int, int, float]:
    """Load sparsity file and compute dead feature statistics."""
    data = st_load_file(str(sparsity_path))
    log_sparsity = data["sparsity"]
    d_sae = log_sparsity.numel()
    dead_mask = log_sparsity < -5
    dead_count = int(dead_mask.sum().item())
    dead_frac = dead_count / d_sae
    return log_sparsity, d_sae, dead_count, dead_frac


def get_held_out_tokens(
    model: Any,
    n_batches: int,
    batch_size: int,
    context_length: int,
    skip_batches: int = 5000,
) -> torch.Tensor:
    """Load held-out tokens from the dataset, skipping training data."""
    ds = load_dataset(
        "NeelNanda/c4-code-tokenized-2b",
        streaming=True,
        split="train",
        trust_remote_code=False,
    )
    ds_iter = iter(ds)
    # Skip past training data
    for _ in range(skip_batches):
        next(ds_iter)

    all_tokens = []
    for _ in range(n_batches):
        row = next(ds_iter)
        toks = row["tokens"][:context_length]
        all_tokens.append(toks)
        if len(all_tokens) >= n_batches * batch_size:
            break

    # Pad/truncate to exact shape
    tokens = torch.tensor(all_tokens[:n_batches], dtype=torch.long, device="cuda")
    return tokens


def compute_feature_activations(
    model: Any,
    sae: SAE,
    tokens: torch.Tensor,
    hook_point: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run tokens through model+SAE, return (max_acts_per_feature, all_feature_acts).

    Returns:
        max_acts: shape (n_tokens_flat, d_sae) - feature activations per token
        tokens_flat: shape (n_tokens_flat,) - corresponding token ids
    """
    all_feature_acts = []
    all_tokens_flat = []

    # Process in small batches to avoid OOM
    batch_size = 8
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i : i + batch_size]
        with torch.inference_mode():
            _, cache = model.run_with_cache(batch, names_filter=[hook_point])
            acts = cache[hook_point]  # (batch, seq, d_model)
            flat_acts = acts.reshape(-1, acts.shape[-1])  # (batch*seq, d_model)
            feature_acts = sae.encode(flat_acts)  # (batch*seq, d_sae)

        all_feature_acts.append(feature_acts.cpu())
        all_tokens_flat.append(batch.reshape(-1).cpu())

        del cache, acts, flat_acts, feature_acts
        torch.cuda.empty_cache()

    return torch.cat(all_feature_acts, dim=0), torch.cat(all_tokens_flat, dim=0)


def find_top_activating_examples(
    feature_acts: torch.Tensor,
    tokens_flat: torch.Tensor,
    all_tokens: torch.Tensor,
    feature_indices: list[int],
    n_top: int,
    context_length: int,
) -> dict[int, list[dict[str, Any]]]:
    """For each feature, find the top-N activating tokens with context."""
    results: dict[int, list[dict[str, Any]]] = {}

    for feat_idx in feature_indices:
        acts = feature_acts[:, feat_idx]
        top_indices = torch.topk(acts, min(n_top, acts.shape[0])).indices

        examples = []
        for idx in top_indices:
            idx = idx.item()
            activation = acts[idx].item()
            if activation <= 0:
                continue

            # Find which sequence and position this token belongs to
            seq_idx = idx // context_length
            pos_idx = idx % context_length
            token_id = tokens_flat[idx].item()

            # Get context window (5 tokens before and after)
            ctx_start = max(0, pos_idx - 5)
            ctx_end = min(context_length, pos_idx + 6)

            context_ids = all_tokens[seq_idx, ctx_start:ctx_end].tolist()
            highlight_pos = pos_idx - ctx_start

            examples.append(
                {
                    "activation": activation,
                    "token_id": token_id,
                    "seq_idx": seq_idx,
                    "pos_idx": pos_idx,
                    "context_ids": context_ids,
                    "highlight_pos": highlight_pos,
                }
            )

        results[feat_idx] = examples

    return results


def generate_html_report(
    hook_point: str,
    dead_count: int,
    d_sae: int,
    dead_frac: float,
    log_sparsity: torch.Tensor,
    top_examples: dict[int, list[dict[str, Any]]],
    tokenizer: Any,
) -> str:
    """Generate an HTML report for the feature analysis."""
    lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>Feature Analysis: {hook_point}</title>",
        "<style>",
        "body { font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; }",
        "h1, h2, h3 { color: #333; }",
        ".stats { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0; }",
        ".feature { border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }",
        ".example { margin: 5px 0; padding: 5px; background: #fafafa; }",
        ".highlight { background: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 3px; }",
        ".act-value { color: #666; font-size: 0.9em; }",
        "</style>",
        "</head><body>",
        f"<h1>Feature Analysis: {html.escape(hook_point)}</h1>",
        "<div class='stats'>",
        f"<p><b>Total features:</b> {d_sae}</p>",
        f"<p><b>Dead features (log_sparsity &lt; -5):</b> {dead_count} ({dead_frac:.1%})</p>",
        f"<p><b>Live features:</b> {d_sae - dead_count}</p>",
        f"<p><b>Mean log sparsity:</b> {log_sparsity.mean().item():.4f}</p>",
        f"<p><b>Features analyzed:</b> {len(top_examples)}</p>",
        "</div>",
    ]

    for feat_idx, examples in sorted(top_examples.items()):
        feat_sparsity = log_sparsity[feat_idx].item()
        lines.append("<div class='feature'>")
        lines.append(
            f"<h3>Feature {feat_idx} "
            f"<span class='act-value'>(log_sparsity: {feat_sparsity:.4f})</span></h3>"
        )

        if not examples:
            lines.append("<p>No activating examples found.</p>")
        else:
            lines.append(f"<p>Top {len(examples)} activating examples:</p>")
            for i, ex in enumerate(examples):
                # Decode context tokens with highlighting
                context_tokens = [tokenizer.decode([tid]) for tid in ex["context_ids"]]
                highlighted = []
                for j, tok_str in enumerate(context_tokens):
                    escaped = html.escape(tok_str)
                    if j == ex["highlight_pos"]:
                        highlighted.append(f"<span class='highlight'>{escaped}</span>")
                    else:
                        highlighted.append(escaped)

                context_html = "".join(highlighted)
                lines.append(
                    f"<div class='example'>"
                    f"<span class='act-value'>[{ex['activation']:.3f}]</span> "
                    f"...{context_html}..."
                    f"</div>"
                )

        lines.append("</div>")

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def analyze_single_sae(
    config_path: Path,
    seed: int,
) -> None:
    """Run feature analysis for a single SAE."""
    cfg = load_config(config_path)
    hook_point = cfg["hook_point"]
    hook_slug = hook_point.replace(".", "_")
    sae_path = Path("outputs") / f"sae_{hook_slug}"

    print(f"\n{'=' * 60}")
    print(f"Analyzing features for: {hook_point}")
    print(f"SAE path: {sae_path}")
    print(f"{'=' * 60}\n")

    # 1. Dead feature statistics
    sparsity_path = sae_path / "sparsity.safetensors"
    log_sparsity, d_sae, dead_count, dead_frac = get_dead_feature_stats(sparsity_path)
    print(f"Dead features: {dead_count}/{d_sae} ({dead_frac:.1%})")

    # 2. Load model and SAE
    model = load_model(device="cuda")
    sae = SAE.load_from_disk(str(sae_path), device="cuda")
    tokenizer = model.tokenizer

    # 3. Get held-out tokens
    print("Loading held-out tokens...")
    tokens = get_held_out_tokens(
        model, N_HELD_OUT_BATCHES, BATCH_SIZE_PROMPTS, CONTEXT_LENGTH
    )
    print(f"Held-out tokens shape: {tokens.shape}")

    # 4. Compute feature activations
    print("Computing feature activations...")
    feature_acts, tokens_flat = compute_feature_activations(
        model, sae, tokens, hook_point
    )
    print(f"Feature activations shape: {feature_acts.shape}")

    # 5. Select random live features
    live_mask = log_sparsity >= -5
    live_indices = live_mask.nonzero(as_tuple=True)[0].tolist()
    n_sample = min(N_SAMPLE_FEATURES, len(live_indices))
    rng = random.Random(seed)
    sampled_features = sorted(rng.sample(live_indices, n_sample))
    print(f"Sampled {n_sample} live features for analysis")

    # 6. Find top activating examples
    print("Finding top activating examples...")
    top_examples = find_top_activating_examples(
        feature_acts,
        tokens_flat,
        tokens,
        sampled_features,
        N_TOP_EXAMPLES,
        CONTEXT_LENGTH,
    )

    # 7. Generate HTML report
    print("Generating report...")
    report_html = generate_html_report(
        hook_point, dead_count, d_sae, dead_frac, log_sparsity, top_examples, tokenizer
    )

    # 8. Save report
    report_dir = Path("outputs") / "feature_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"features_{hook_slug}.html"
    report_path.write_text(report_html)
    print(f"Report saved to: {report_path}")

    # 9. Log to MLflow
    init_mlflow()
    with start_run(
        run_name=f"feature_analysis_{hook_slug}",
        hook_point=hook_point,
        run_purpose="feature_analysis",
    ):
        mlflow.log_params(
            {
                "hook_point": hook_point,
                "d_sae": d_sae,
                "n_sample_features": n_sample,
                "n_top_examples": N_TOP_EXAMPLES,
                "n_held_out_batches": N_HELD_OUT_BATCHES,
                "seed": seed,
            }
        )
        mlflow.log_metrics(
            {
                "dead_feature_count": dead_count,
                "dead_feature_fraction": dead_frac,
                "live_feature_count": d_sae - dead_count,
                "mean_log_sparsity": log_sparsity.mean().item(),
            }
        )
        mlflow.log_artifact(str(report_path), "feature_reports")

    print(f"MLflow run logged for {hook_point}")

    # Cleanup
    del model, sae, feature_acts, tokens_flat, tokens
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAE features")
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to YAML config(s)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project requires a CUDA GPU.")

    set_seed(args.seed)

    for config_path in args.config:
        analyze_single_sae(config_path, args.seed)

    print("\nFeature analysis complete for all hook points.")


if __name__ == "__main__":
    main()
