"""Evaluation script for trained Activation Oracles."""

from __future__ import annotations

import argparse
import html
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel

from src.data import load_classification_dataset, make_training_example
from src.model import (
    _resolve_layers,
    extract_activation,
    get_activation_layer,
    get_injection_layer,
    load_base_model,
    load_tokenizer,
    make_steering_hook,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_oracle(model_name: str, lora_path: str, device: str = "cuda") -> Any:
    base_model = load_base_model(model_name, device=device)
    oracle = PeftModel.from_pretrained(base_model, lora_path)
    oracle.eval()
    return oracle, base_model


def run_eval(
    target_model,
    oracle_model,
    tokenizer,
    eval_examples: list[dict],
    activation_depth: float,
    n_placeholders: int = 1,
    max_examples: int | None = None,
) -> list[dict]:
    """Run evaluation and return per-example results."""
    results = []
    injection_layer = get_injection_layer(oracle_model)
    activation_layer = get_activation_layer(target_model, activation_depth)
    placeholder_id = tokenizer.encode(" ?", add_special_tokens=False)[0]
    layers = _resolve_layers(oracle_model)

    examples = eval_examples[:max_examples] if max_examples else eval_examples

    for ex in examples:
        activation = extract_activation(
            target_model, tokenizer, ex["text"], depth_frac=activation_depth
        )

        item = make_training_example(
            ex["text"], ex["label"], tokenizer,
            activation_layer=activation_layer,
            n_placeholders=n_placeholders,
        )

        ids = item["input_ids"].unsqueeze(0).to(oracle_model.device)
        pos = (ids[0] == placeholder_id).nonzero(as_tuple=True)[0]
        pp = pos.tolist() if pos.numel() > 0 else None

        hook = make_steering_hook(activation, pp)
        handle = layers[injection_layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                gen_ids = oracle_model.generate(
                    ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            handle.remove()

        answer = tokenizer.decode(
            gen_ids[0, item["answer_start_pos"]:], skip_special_tokens=True
        ).strip().lower()
        expected = "yes" if ex["label"] == 1 else "no"
        correct = expected in answer

        results.append({
            "text": ex["text"],
            "label": ex["label"],
            "expected": expected,
            "predicted": answer,
            "correct": correct,
        })

    return results


def generate_report(results: list[dict], model_name: str, lora_path: str) -> str:
    """Generate HTML evaluation report."""
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / max(total, 1)

    lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Activation Oracle Evaluation</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}",
        "h1{color:#333;}",
        ".summary{background:#f5f5f5;padding:15px;border-radius:5px;margin:15px 0;}",
        ".result{margin:8px 0;padding:10px;border-radius:4px;}",
        ".correct{background:#e8f5e9;border-left:4px solid #4caf50;}",
        ".incorrect{background:#ffebee;border-left:4px solid #f44336;}",
        ".text{font-style:italic;color:#555;}",
        ".pred{font-family:monospace;}",
        ".label-correct{color:#2e7d32;font-weight:bold;}",
        ".label-incorrect{color:#c62828;font-weight:bold;}",
        "</style>",
        "</head><body>",
        "<h1>Activation Oracle Evaluation</h1>",
        "<div class='summary'>",
        f"<p><b>Model:</b> {html.escape(model_name)}</p>",
        f"<p><b>LoRA adapter:</b> {html.escape(lora_path)}</p>",
        "<p><b>Task:</b> Sentiment classification (SST-2)</p>",
        f"<p><b>Accuracy:</b> {correct}/{total} ({accuracy:.1%})</p>",
        "</div>",
        "<h2>Sample Results</h2>",
    ]

    for r in results[:50]:
        cls = "correct" if r["correct"] else "incorrect"
        lbl_cls = "label-correct" if r["correct"] else "label-incorrect"
        lines.append(f"<div class='result {cls}'>")
        lines.append(f"<div class='text'>{html.escape(r['text'][:120])}</div>")
        lines.append("<div class='pred'>")
        lines.append(f"Expected: <span class='{lbl_cls}'>{r['expected']}</span> | ")
        lines.append(f"Predicted: {html.escape(r['predicted'][:100])}")
        lines.append("</div></div>")

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Activation Oracle")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    lora_path = str(args.lora_path)

    print(f"Loading model {model_name} with LoRA from {lora_path}")
    tokenizer = load_tokenizer(model_name)
    oracle, target = load_oracle(model_name, lora_path)

    print("Loading eval dataset...")
    _, eval_ex = load_classification_dataset(
        dataset_name=cfg.get("dataset_name", "stanfordnlp/sst2"),
        max_train=0,
        max_eval=cfg.get("max_eval_samples", 500),
    )

    print(f"Running eval on {len(eval_ex)} examples...")
    results = run_eval(
        target_model=target,
        oracle_model=oracle,
        tokenizer=tokenizer,
        eval_examples=eval_ex,
        activation_depth=cfg["activation_depth"],
        max_examples=args.max_eval,
    )

    accuracy = sum(1 for r in results if r["correct"]) / max(len(results), 1)
    print(f"\nAccuracy: {accuracy:.3f} ({sum(1 for r in results if r['correct'])}/{len(results)})")

    report = generate_report(results, model_name, lora_path)

    if args.output:
        output_path = args.output
    else:
        output_path = Path("outputs") / "eval_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")

    del oracle, target, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
