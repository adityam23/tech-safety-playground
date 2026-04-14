"""Training entry point for sparse autoencoders. Takes --config and --seed."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from safetensors.torch import load_file as st_load_file
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    SAE,
)
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer, TrainStepOutput

from src.data import extract_activations, load_model
from src.mlflow_utils import init_mlflow, log_params_dict, start_run


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


def build_sae_lens_config(
    cfg: dict[str, Any], seed: int, output_path: Path
) -> LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]:
    """Build SAELens training config from our YAML config."""
    sae_config = StandardTrainingSAEConfig(
        d_in=cfg["d_model"],
        d_sae=cfg["d_sae"],
        l1_coefficient=cfg["l1_coefficient"],
        l1_warm_up_steps=cfg.get("l1_warm_up_steps", 0),
        dtype="float32",
        device="cuda",
    )

    runner_config = LanguageModelSAERunnerConfig(
        sae=sae_config,
        model_name=cfg["model_name"],
        hook_name=cfg["hook_point"],
        dataset_path="NeelNanda/c4-code-tokenized-2b",
        is_dataset_tokenized=True,
        streaming=True,
        context_size=cfg["context_length"],
        training_tokens=cfg["total_training_tokens"],
        train_batch_size_tokens=cfg["batch_size"],
        # Activation store: keep buffer small for 6 GB VRAM
        store_batch_size_prompts=16,
        n_batches_in_buffer=64,
        # Optimizer
        lr=cfg["learning_rate"],
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Reproducibility
        seed=seed,
        device="cuda",
        prepend_bos=True,
        # Checkpointing
        n_checkpoints=5,
        save_final_checkpoint=True,
        output_path=str(output_path),
        # Log to both wandb and MLflow
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="sae-lens-resid-strm",
            wandb_log_frequency=100,
        ),
    )

    return runner_config


class MLflowTrainingRunner(LanguageModelSAETrainingRunner):
    """Subclass that patches the trainer to also log step metrics to MLflow."""

    def run(self) -> Any:
        """Override run to patch the trainer before training starts."""
        # Monkey-patch SAETrainer.__init__ to intercept trainer creation
        original_init = SAETrainer.__init__

        def patched_init(trainer_self: SAETrainer, *args: Any, **kwargs: Any) -> None:
            original_init(trainer_self, *args, **kwargs)
            _patch_trainer_for_mlflow(trainer_self)

        SAETrainer.__init__ = patched_init  # type: ignore[method-assign]
        try:
            return super().run()
        finally:
            SAETrainer.__init__ = original_init  # type: ignore[method-assign]


def _patch_trainer_for_mlflow(trainer: SAETrainer) -> None:
    """Monkey-patch the SAETrainer to also log step metrics to MLflow."""
    original_log = trainer._log_train_step

    @torch.no_grad()
    def _log_train_step_mlflow(step_output: TrainStepOutput) -> None:
        # Call original wandb logging
        original_log(step_output)

        # Mirror to MLflow on the same schedule as wandb
        if not trainer._is_logging_step():
            return

        step = trainer.n_training_steps
        log_dict = trainer._build_train_step_log_dict(
            output=step_output,
            n_training_samples=trainer.n_training_samples,
        )

        # Log key metrics to MLflow (skip wandb Histogram objects)
        mlflow.log_metrics(
            {
                "losses/mse_loss": log_dict.get("losses/mse_loss", 0.0),
                "losses/l1_loss": log_dict.get("losses/l1", 0.0),
                "losses/overall_loss": log_dict["losses/overall_loss"],
                "metrics/l0": log_dict["metrics/l0"],
                "metrics/explained_variance": log_dict["metrics/explained_variance"],
                "sparsity/dead_features": log_dict["sparsity/dead_features"],
                "sparsity/mean_passes_since_fired": log_dict[
                    "sparsity/mean_passes_since_fired"
                ],
            },
            step=step,
        )

    trainer._log_train_step = _log_train_step_mlflow


def log_final_metrics(output_path: Path, hook_point: str) -> None:
    """Compute and log final metrics from saved SAE and sparsity files."""
    # Load sparsity stats
    sparsity_path = output_path / "sparsity.safetensors"
    if sparsity_path.exists():
        sparsity_data = st_load_file(str(sparsity_path))
        log_sparsity = sparsity_data["sparsity"]
        d_sae = log_sparsity.numel()

        dead_mask = log_sparsity < -5
        dead_count = int(dead_mask.sum().item())
        dead_frac = dead_count / d_sae

        mlflow.log_metrics(
            {
                "final/dead_feature_count": dead_count,
                "final/dead_feature_fraction": dead_frac,
                "final/total_features": d_sae,
                "final/mean_log_feature_sparsity": log_sparsity.mean().item(),
            }
        )
        print(
            f"Dead features (log_sparsity < -5): {dead_count}/{d_sae} ({dead_frac:.1%})"
        )

    # Load SAE and compute L0 on a sample
    sae = SAE.load_from_disk(str(output_path), device="cuda")
    model = load_model(device="cuda")
    tokens = torch.randint(0, model.cfg.d_vocab, (8, 128), device="cuda")
    acts = extract_activations(model, tokens, hook_points=[hook_point])
    x = acts[hook_point].reshape(-1, model.cfg.d_model)

    with torch.inference_mode():
        encoded = sae.encode(x)
        l0 = (encoded > 0).float().sum(dim=-1).mean().item()

    mlflow.log_metric("final/l0", l0)
    print(f"Final L0 (sample): {l0:.1f}")

    del sae, model, tokens, acts, x, encoded
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAE on gelu-2l residual stream")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Validate CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project requires a CUDA GPU.")

    # Load config and set seed
    cfg = load_config(args.config)
    seed = args.seed
    set_seed(seed)

    # Build output path from hook point name
    hook_slug = cfg["hook_point"].replace(".", "_")
    output_path = Path("outputs") / f"sae_{hook_slug}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Build SAELens config
    sae_lens_cfg = build_sae_lens_config(cfg, seed, output_path)

    # Save full config as JSON for reproducibility
    config_record = {**cfg, "seed": seed, "output_path": str(output_path)}
    config_json_path = output_path / "config.json"
    with open(config_json_path, "w") as f:
        json.dump(config_record, f, indent=2)

    # Initialize MLflow and start run
    init_mlflow()
    with start_run(
        run_name=f"train_{hook_slug}",
        hook_point=cfg["hook_point"],
        run_purpose="initial_training",
    ):
        # Log all params
        log_params_dict(
            {
                "model_name": cfg["model_name"],
                "hook_point": cfg["hook_point"],
                "d_model": cfg["d_model"],
                "d_sae": cfg["d_sae"],
                "expansion_factor": cfg["expansion_factor"],
                "l1_coefficient": cfg["l1_coefficient"],
                "l1_warm_up_steps": cfg.get("l1_warm_up_steps", 0),
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "context_length": cfg["context_length"],
                "total_training_tokens": cfg["total_training_tokens"],
                "optimizer": cfg["optimizer"],
                "seed": seed,
                "activation_function": cfg["activation_function"],
                "normalization": cfg["normalization"],
                "store_batch_size_prompts": 16,
                "n_batches_in_buffer": 64,
            }
        )

        # Log config JSON as artifact
        mlflow.log_artifact(str(config_json_path), "config")

        # Train
        print(f"Starting SAE training at hook point: {cfg['hook_point']}")
        print(f"Training tokens: {cfg['total_training_tokens']:,}")
        print(
            f"d_model={cfg['d_model']}, d_sae={cfg['d_sae']}, l1={cfg['l1_coefficient']}"
        )
        print(f"Seed: {seed}")
        print()

        torch.cuda.reset_peak_memory_stats()
        runner = MLflowTrainingRunner(cfg=sae_lens_cfg)
        sae = runner.run()

        # Log peak VRAM
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        mlflow.log_metric("peak_vram_mb", peak_vram_mb)
        print(f"\nPeak VRAM: {peak_vram_mb:.1f} MB")

        # Compute and log final metrics from saved files
        log_final_metrics(output_path, cfg["hook_point"])

        # Log saved SAE weights and artifacts
        if output_path.exists():
            mlflow.log_artifacts(str(output_path), "sae_weights")

        print(f"SAE weights saved to: {output_path}")
        print("MLflow run logged successfully.")

    # Cleanup
    del sae, runner
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
