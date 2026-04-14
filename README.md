# Sparse Autoencoder on gelu-2l Residual Stream

Mechanistic interpretability project: train sparse autoencoders (SAEs) on the residual stream of **gelu-2l** (a 2-layer transformer) at three hook points, analyze learned features, and demonstrate activation steering.

## Setup

```bash
# Clone and enter the repo
git clone <repo-url> && cd techsafety

# Install dependencies (requires uv)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv sync

# Verify GPU and run smoke tests
uv run pytest tests/ -v
```

**Requirements:** NVIDIA GPU with 6 GB VRAM, CUDA 12.8, Python 3.14+, `uv`.

## MLflow

All runs are tracked locally via MLflow.

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000.

## Project Phases

1. **Scaffolding** — repo structure, smoke test
2. **Train SAE** at `blocks.0.hook_resid_post`
3. **Train SAEs** at the other two hook points
4. **Feature analysis** — dead features, top activations, reports
5. **Steering demo** — amplify/suppress interpretable features
6. **Writeup** — this README with findings

## Hook Points

| Hook Point | Description |
|---|---|
| `blocks.0.hook_resid_pre` | Input to layer 0 |
| `blocks.0.hook_resid_post` | Between blocks 0 and 1 |
| `blocks.1.hook_resid_post` | Output of layer 1 |

## Key References

- [SAELens](https://github.com/decoderesearch/SAELens)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Anthropic, "Towards Monosemanticity" (Bricken et al., 2023)
- Anthropic, "Scaling Monosemanticity" (2024)
