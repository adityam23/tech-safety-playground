"""Training entry point for sparse autoencoders. Takes --config and --seed."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAE on gelu-2l residual stream")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # TODO: Phase 2 implementation
    print(f"Config: {args.config}, Seed: {args.seed}")
    print("Training not yet implemented — see Phase 2.")


if __name__ == "__main__":
    main()
