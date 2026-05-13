# AI Safety Playground

A monorepo of AI safety experiments: mechanistic interpretability, model control, and explainable AI. Each project explores a different technique for understanding or steering model behavior.

## Setup

Each project has its own dependencies managed by [`uv`](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/adityam23/tech-safety-playground.git
cd tech-safety-playground
```

Then `uv sync` inside any project directory to install its dependencies. Use `--recurse-submodules` when cloning so the submodule projects are checked out automatically.

## Projects

### [sparse-autoencoders](projects/sparse-autoencoders/)

Sparse autoencoders trained on the gelu-2l residual stream at three hook points. Includes feature analysis, top-activating examples, and activation steering with a coherence/steering strength sweep.

**Setup:** `cd projects/sparse-autoencoders && uv sync`

### [explainable-crowdfunding-ml](https://github.com/adityam23/explainable-crowdfunding-ml)

Explainable AI analysis of Kickstarter crowdfunding data using SHAP, LIME (with DICE variant), and XGBoost. Replicates and extends a published study on sustainability-oriented crowdfunding. Tracked as a git submodule.

**Setup:** `cd projects/explainable-crowdfunding-ml && uv sync`

### [activation-oracles](projects/activation-oracles/)

Activation Oracles (Karvonen et al., Dec 2025) -- train an LLM to accept its own residual stream activations and answer natural-language questions about them. Qwen2.5-0.5B as both target and oracle, LoRA-trained on SST-2 sentiment classification as a proof of concept.

**Setup:** `cd projects/activation-oracles && uv sync`
