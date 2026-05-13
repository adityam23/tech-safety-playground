# Sparse Autoencoder on gelu-2l Residual Stream

Train sparse autoencoders (SAEs) on the residual stream of **gelu-2l** -- a 2-layer, 512-dimensional transformer trained on code and natural language -- at three hook points, analyze the learned features for interpretability, and demonstrate activation steering with a strength sweep that characterizes the coherence/steering tradeoff.

## Setup

**Requirements:** NVIDIA GPU with >=6 GB VRAM, CUDA 12.8+, Python 3.14+, [`uv`](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/adityam23/tech-safety-playground.git
cd tech-safety-playground/projects/sparse-autoencoders
uv sync
uv run pytest tests/ -v
```

PyTorch was installed with CUDA support via:
```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Reproduction

```bash
# Train SAEs at all three hook points
PYTHONPATH=. uv run python src/train_sae.py --config configs/sae_resid_post_l0.yaml
PYTHONPATH=. uv run python src/train_sae.py --config configs/sae_resid_pre_l0.yaml
PYTHONPATH=. uv run python src/train_sae.py --config configs/sae_resid_post_l1.yaml

# Feature analysis (all three SAEs at once)
PYTHONPATH=. uv run python src/analyze_features.py --config configs/sae_resid_post_l0.yaml configs/sae_resid_pre_l0.yaml configs/sae_resid_post_l1.yaml

# Steering demo + strength sweep
PYTHONPATH=. uv run python src/steering_demo.py --config configs/sae_resid_post_l0.yaml
PYTHONPATH=. uv run python src/steering_sweep.py --config configs/sae_resid_post_l0.yaml

# View results
mlflow ui --backend-store-uri ./mlruns
```

Reports are saved as HTML in `outputs/feature_reports/` and `outputs/steering_reports/`. MLflow at http://localhost:5000. Experiment: `sae-gelu2l-residual`.

## Training Results

Each SAE was trained for 100M tokens on `NeelNanda/c4-code-tokenized-2b` with expansion factor 8x (d_sae = 4096). Training used SAELens with Adam optimizer, LR 3e-4, batch size 4096, context length 128, and 2000-step L1 warmup.

| Hook Point | Position | L1 Coeff | L0 | Dead Features | Dead % | Mean Log Sparsity |
|---|---|---|---|---|---|---|
| `blocks.0.hook_resid_pre` | Input to layer 0 | 1.0 | 18.4 | 19 / 4096 | 0.5% | -3.28 |
| `blocks.0.hook_resid_post` | Between blocks | 3.0 | 29.3 | 7 / 4096 | 0.2% | -2.91 |
| `blocks.1.hook_resid_post` | Output of layer 1 | 50.0 | 17.6 | 336 / 4096 | 8.2% | -3.66 |

**Key observation: per-layer L1 tuning.** The three hook points required very different L1 coefficients (1.0, 3.0, 50.0) because activation magnitudes increase with depth. The final layer has roughly 10x larger activations than the first. L1 warmup over 2000 steps was essential.

## Feature Analysis

For each SAE, we computed dead-feature statistics and found the top-20 activating examples for 50 randomly sampled live features on held-out data.

**Example interpretable features (from `blocks.0.hook_resid_post`)**

- **Feature 26 -- Frequency adverbs:** "generally", "usually", "commonly", "often"
- **Feature 131 -- Tech products/brands:** "Apple Watch", "Galaxy", "Xiaomi"
- **Feature 357 -- Code variable patterns:** comma-separated variable names (i, j, k)
- **Feature 420 -- Money/financial:** "money", "dollars", financial contexts
- **Feature 384 -- Reference/regard words:** "reference", "respect", "regard"

## Steering Demo

Activation steering uses SAE decoder weights as concept directions. Applied via a TransformerLens hook on `blocks.0.hook_resid_post` during autoregressive generation (temperature 0.8, top-k 50, 60 new tokens).

**Results at strength=20:** Features 420 (money) and 131 (tech brands) showed convincing steering. Features 26, 103, and 384 collapsed into degenerate text, indicating strength=20 is past their coherence cliff.

## Steering Strength Sweep

| Feature | Coherent through | Steering visible at | Degenerates at | Useful range |
|---|---|---|---|---|
| 420 (money) | str ~ 10 | str = 5 | str >= 15 | **5-10** |
| 131 (tech brands) | str ~ 5 | str = 5 | str >= 10 | **5** |
| 26 (adverbs) | str ~ 2 | str = 2 | str >= 5 | **2** |
| 103 (need to) | str ~ 2 | str = 2 | str >= 5 | **2** |
| 384 (reference) | str ~ 2 | str = 5 | str >= 10 | **2-5** |

**Findings:** Feature-specific optimal strengths exist. Semantic features are more robust than syntactic ones. The coherence cliff is sharp, not gradual.

## Hook Points

| Hook Point | Description |
|---|---|
| `blocks.0.hook_resid_pre` | Input to layer 0 (embedding output) |
| `blocks.0.hook_resid_post` | Between blocks 0 and 1 |
| `blocks.1.hook_resid_post` | Output of layer 1 (final residual stream) |

## Limitations

Single model (gelu-2l), single SAE architecture (ReLU, 8x), single seed (42), 100M training tokens, no automated interpretability, 6 GB VRAM constraint.

## References

- [SAELens](https://github.com/decoderesearch/SAELens) -- SAE training library
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) -- mechanistic interpretability toolkit
- Bricken et al., ["Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Anthropic, 2023)
- Templeton et al., ["Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, 2024)
