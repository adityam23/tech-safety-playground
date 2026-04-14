"""Feature analysis: dead features, top activations, feature categorization."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAE features")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # TODO: Phase 4 implementation
    print(f"Config: {args.config}, Seed: {args.seed}")
    print("Feature analysis not yet implemented — see Phase 4.")


if __name__ == "__main__":
    main()
