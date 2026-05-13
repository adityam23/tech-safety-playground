# Tech Safety Playground

Experiments in mechanistic interpretability, model control, and AI alignment. Each project in this repo explores a different technique for understanding or steering model behavior.

## Setup

Each project has its own dependencies managed by [`uv`](https://docs.astral.sh/uv/). Clone the repo and `uv sync` in any project directory.

```bash
git clone <repo-url> && cd techsafety
```

## Projects

### [sparse-autoencoders](projects/sparse-autoencoders/)

Sparse autoencoders trained on the gelu-2l residual stream at three hook points. Includes feature analysis, top-activating examples, and activation steering with a coherence/steering strength sweep.

**Setup:** `cd projects/sparse-autoencoders && uv sync`

### [explainable-crowdfunding-ml](projects/explainable-crowdfunding-ml/)

Explainable AI analysis of Kickstarter crowdfunding data using SHAP, LIME (with DICE variant), and XGBoost. Replicates and extends a published study on sustainability-oriented crowdfunding.

**Setup:** `cd projects/explainable-crowdfunding-ml && uv sync`
