"""Steering demo: amplify/suppress features and generate text."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SAE steering demo")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # TODO: Phase 5 implementation
    print(f"Config: {args.config}, Seed: {args.seed}")
    print("Steering demo not yet implemented — see Phase 5.")


if __name__ == "__main__":
    main()
