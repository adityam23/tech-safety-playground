"""Training loop for Activation Oracles -- LoRA fine-tune on classification."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup

from src.data import load_classification_dataset, make_training_example
from src.model import (
    _resolve_layers,
    apply_lora,
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


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


class AODataset(Dataset):
    """Dataset that pre-tokenizes oracle prompts for classification training."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        activation_layer: int,
        n_placeholders: int = 1,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.activation_layer = activation_layer
        self.n_placeholders = n_placeholders

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        item = make_training_example(
            text=ex["text"],
            label=ex["label"],
            tokenizer=self.tokenizer,
            activation_layer=self.activation_layer,
            n_placeholders=self.n_placeholders,
        )
        return {
            "target_text": item["target_text"],
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "answer_start_pos": item["answer_start_pos"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences in a batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    labels = []
    answer_starts = []

    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        ids = torch.cat([item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)])
        lbls = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(lbls)
        answer_starts.append(item["answer_start_pos"])

    return {
        "target_text": [item["target_text"] for item in batch],
        "input_ids": torch.stack(input_ids),   # (batch, seq_len)
        "labels": torch.stack(labels),          # (batch, seq_len)
        "answer_start_pos": answer_starts,
    }


def train_step(
    target_model: nn.Module,
    oracle_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    batch: dict,
    injection_layer: int,
    activation_depth: float,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    placeholder_token: str = " ?",
) -> float:
    """One training step: extract activations, inject them into oracle, compute loss."""
    device = oracle_model.device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    target_texts = batch["target_text"]

    placeholder_id = tokenizer.encode(placeholder_token, add_special_tokens=False)[0]

    losses = []
    for i in range(input_ids.shape[0]):
        text = target_texts[i]
        ids = input_ids[i:i+1]  # (1, seq_len)
        lbl = labels[i:i+1]

        # Extract activation from target model (LoRA disabled)
        activation = extract_activation(
            target_model, tokenizer, text, depth_frac=activation_depth
        )

        # Find placeholder positions in this example
        pos = (ids[0] == placeholder_id).nonzero(as_tuple=True)[0]
        placeholder_positions = pos.tolist() if pos.numel() > 0 else None

        # Run oracle forward with injection at layer 1
        steer_hook_fn = make_steering_hook(activation, placeholder_positions)
        handle = _resolve_layers(oracle_model)[injection_layer].register_forward_hook(
            steer_hook_fn
        )
        try:
            outputs = oracle_model(input_ids=ids, labels=lbl)
            loss = outputs.loss
        finally:
            handle.remove()

        losses.append(loss)

    total_loss = torch.stack(losses).mean()
    total_loss.backward()

    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_(oracle_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss.item()


def evaluate(
    target_model: nn.Module,
    oracle_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    eval_examples: list[dict],
    injection_layer: int,
    activation_depth: float,
    n_placeholders: int = 1,
    max_batches: int | None = None,
) -> float:
    """Evaluate classification accuracy on held-out examples."""
    oracle_model.eval()
    correct = 0
    total = 0
    placeholder_token = " ?"
    placeholder_id = tokenizer.encode(placeholder_token, add_special_tokens=False)[0]

    for ex in eval_examples[:max_batches]:
        text = ex["text"]
        label = ex["label"]

        item = make_training_example(
            text, label, tokenizer,
            activation_layer=get_activation_layer(target_model, activation_depth),
            n_placeholders=n_placeholders,
        )

        # Extract activation
        activation = extract_activation(
            target_model, tokenizer, text, depth_frac=activation_depth
        )

        # Generate answer with steering
        ids = item["input_ids"].unsqueeze(0).to(oracle_model.device)
        pos = (ids[0] == placeholder_id).nonzero(as_tuple=True)[0]
        placeholder_positions = pos.tolist() if pos.numel() > 0 else None

        steer_hook_fn = make_steering_hook(activation, placeholder_positions)
        handle = _resolve_layers(oracle_model)[injection_layer].register_forward_hook(
            steer_hook_fn
        )
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

        answer_text = tokenizer.decode(
            gen_ids[0, item["answer_start_pos"]:], skip_special_tokens=True
        ).strip().lower()

        expected = "yes" if label == 1 else "no"
        if expected in answer_text:
            correct += 1
        total += 1

    oracle_model.train()
    return correct / max(total, 1)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train Activation Oracle")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    cfg = load_config(args.config)
    seed = cfg.get("seed", args.seed)
    set_seed(seed)

    device = cfg["device"]
    dtype = torch.bfloat16

    print(f"Loading model: {cfg['model_name']}")
    tokenizer = load_tokenizer(cfg["model_name"])
    base_model = load_base_model(cfg["model_name"], device=device, dtype=dtype)

    injection_layer = get_injection_layer(base_model)
    activation_layer = get_activation_layer(base_model, cfg["activation_depth"])
    print(f"Activation layer (target): {activation_layer}")
    print(f"Injection layer (oracle): {injection_layer}")

    # Apply LoRA to base model → oracle
    oracle_model = apply_lora(base_model, cfg)
    oracle_model.train()

    # Target model: same base, LoRA disabled for activation extraction
    target_model = base_model  # same model, LoRA will be disabled during extraction

    # Load dataset
    print(f"Loading dataset: {cfg['dataset_name']}")
    train_ex, eval_ex = load_classification_dataset(
        dataset_name=cfg["dataset_name"],
        max_train=cfg.get("max_train_samples"),
        max_eval=cfg.get("max_eval_samples"),
        seed=seed,
    )
    print(f"Train examples: {len(train_ex)}, Eval examples: {len(eval_ex)}")

    # Create dataloader
    ds = AODataset(
        train_ex, tokenizer,
        activation_layer=activation_layer,
        n_placeholders=1,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer (only LoRA params are trainable)
    trainable_params = [p for p in oracle_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    total_steps = len(loader) * cfg["num_train_epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # MLflow
    mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")
    mlflow.set_experiment("activation-oracles")
    with mlflow.start_run(run_name="ao_qwen05b_sst2"):
        mlflow.log_params({k: str(v) for k, v in cfg.items()})

        global_step = 0
        accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

        print(f"\nStarting training: {len(loader)} steps/epoch, "
              f"{len(loader) // accumulation_steps} optimizer steps/epoch")
        print(f"Total training tokens: ~{len(train_ex) * 50:,}")

        for epoch in range(cfg["num_train_epochs"]):
            epoch_loss = 0.0
            for step, batch in enumerate(loader):
                loss = train_step(
                    target_model=target_model,
                    oracle_model=oracle_model,
                    tokenizer=tokenizer,
                    batch=batch,
                    injection_layer=injection_layer,
                    activation_depth=cfg["activation_depth"],
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                epoch_loss += loss
                global_step += 1

                if global_step % 100 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    print(f"  Step {global_step:5d} | loss: {loss:.4f} | avg: {avg_loss:.4f}")
                    mlflow.log_metrics(
                        {"train/loss": loss, "train/avg_loss": avg_loss},
                        step=global_step,
                    )

            avg_epoch_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")

            # Evaluate
            acc = evaluate(
                target_model=target_model,
                oracle_model=oracle_model,
                tokenizer=tokenizer,
                eval_examples=eval_ex,
                injection_layer=injection_layer,
                activation_depth=cfg["activation_depth"],
                max_batches=200,
            )
            print(f"Eval accuracy: {acc:.3f}")
            mlflow.log_metrics({"eval/accuracy": acc}, step=global_step)

        # Save LoRA adapter
        output_dir = Path(cfg.get("output_dir", "outputs/ao_qwen05b"))
        output_dir.mkdir(parents=True, exist_ok=True)
        oracle_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifacts(str(output_dir), "lora_adapter")
        print(f"LoRA adapter saved to: {output_dir}")

    del target_model, oracle_model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
