# CLAUDE.md — Sparse Autoencoder on gelu-2l Residual Stream

## Project goal

Train sparse autoencoders (SAEs) on the residual stream of **gelu-2l** (a 2-layer transformer trained by Neel Nanda, available via TransformerLens) at three hook points, then analyze the learned features for interpretability and demonstrate basic activation steering.

This is a mechanistic interpretability project for an AI safety course. The scope is **"standard"**: one SAE per hook point, dead-feature statistics, top-activating examples, and a basic steering demo. No architecture comparisons in this iteration — keep things focused.

## Core experiment

**Model:** `gelu-2l` via TransformerLens (`HookedTransformer.from_pretrained("gelu-2l")`).

**Hook points (train one SAE per point):**
- `blocks.0.hook_resid_pre` — input to layer 0
- `blocks.0.hook_resid_post` — between the two blocks
- `blocks.1.hook_resid_post` — output of layer 1

**SAE:** Standard ReLU sparse autoencoder with L1 penalty on hidden activations. Start with expansion factor 8× (d_sae = 8 * d_model). Use SAELens for training — do not reimplement from scratch.

**Training data:** Neel Nanda's `NeelNanda/c4-code-tokenized-2b` or equivalent standard token dataset used in SAELens tutorials. Start with ~100M–300M tokens for the first run; this should take ~1–2 hours on a single GPU or M-series Mac.

## Environment and dependencies

- **Package manager: `uv`.** Do not use `pip`, `venv`, `conda`, or `poetry` directly. All dependency and environment management goes through `uv`.
- Use the system's installed Python directly via `uv` — do not run `uv python install` or have `uv` download its own Python. Set `requires-python = ">=3.12"` (or whatever matches the system Python) in `pyproject.toml`.
- Initialize the project with `uv init` and let `uv` create the venv automatically on the first `uv add` / `uv run`. Add dependencies via `uv add <pkg>` (never edit `pyproject.toml` dependency lists by hand unless resolving a conflict).
- Run all scripts via `uv run python src/...` so the project venv is used consistently.
- Lockfile: `uv.lock` must be committed.
- Key libraries: `sae-lens`, `transformer-lens`, `torch`, `datasets`, `mlflow`, `plotly`, `pandas`, `numpy`.

## Hardware — local GPU, 6 GB VRAM

**Target hardware: a single local NVIDIA GPU with 6 GB VRAM.** All training, analysis, and the steering demo must fit within this budget. Do not write code that assumes more memory.

- **Device selection:** always use CUDA. Fail fast with a clear error if CUDA is unavailable — do not silently fall back to CPU or MPS for this project.
- **Install the correct PyTorch build for CUDA** via `uv` using the PyTorch index. Document the exact `uv add` command in the README (e.g., `uv add torch --index https://download.pytorch.org/whl/cu124` or the current stable CUDA wheel index at project start).
- **Memory budget rules:**
  - gelu-2l itself is tiny (<100 MB); the SAE at 8× expansion is also small. The memory pressure comes from activation buffers and batch size.
  - Default training batch size (tokens per step): start at **4096 tokens**, context length **128**. Increase only if `nvidia-smi` shows <4 GB used; decrease if OOM.
  - Activation store / buffer size in SAELens: cap at **~256k tokens** in the buffer at once. The SAELens default is much larger and will OOM on 6 GB.
  - Use `torch.float32` for SAE weights (stability matters more than speed here), but store cached activations in `float16` on disk if you cache them.
  - Always wrap inference-only analysis (feature analysis, steering) in `torch.inference_mode()`.
  - Call `torch.cuda.empty_cache()` between phases of the analysis scripts.
- **Checkpoint:** after Phase 1's smoke test, log peak VRAM usage (`torch.cuda.max_memory_allocated()`) to MLflow as a sanity check. If peak exceeds 5 GB during a short training step, shrink batch/buffer before proceeding.

## Experiment tracking — MLflow (local)

**All runs must be tracked with MLflow, running locally.**

- Tracking URI: `file:./mlruns` (local filesystem, no server setup)
- Experiment name: `sae-gelu2l-residual`
- One MLflow run per (hook_point, architecture, seed) combination
- Start the MLflow UI with `mlflow ui --backend-store-uri ./mlruns` and document this in the README

**Log the following for every run:**

*Params:* model name, hook point, d_model, d_sae, expansion factor, l1 coefficient, learning rate, batch size, total training tokens, optimizer, seed, activation function, normalization scheme.

*Metrics (logged per training step or every N steps):* reconstruction loss (MSE), L1 loss, total loss, L0 (average number of active features per token), explained variance, dead feature count, dead feature fraction.

*Artifacts:* trained SAE weights (as `.pt` or via SAELens's save format), a `config.json` with the full run config, the top-activating-examples HTML/markdown report, steering demo outputs, any feature visualization plots.

*Tags:* git commit SHA, hook_point, run_purpose (e.g., `initial_training`, `steering_demo`, `feature_analysis`).

## Repository structure

```
.
├── CLAUDE.md                    # this file
├── README.md                    # human-facing project description + MLflow UI instructions
├── pyproject.toml               # managed by uv
├── uv.lock                      # committed
├── .python-version              # written by uv
├── configs/
│   ├── sae_resid_pre_l0.yaml
│   ├── sae_resid_post_l0.yaml
│   └── sae_resid_post_l1.yaml
├── src/
│   ├── __init__.py
│   ├── train_sae.py             # training entry point, takes --config
│   ├── analyze_features.py      # dead features, top activations, feature categorization
│   ├── steering_demo.py         # amplify/suppress features, generate text
│   ├── data.py                  # dataset loading and activation caching
│   ├── mlflow_utils.py          # wrappers for consistent logging
│   └── viz.py                   # plotting helpers
├── notebooks/
│   └── exploration.ipynb        # interactive exploration; do not put core logic here
├── tests/
│   └── test_smoke.py            # fast smoke tests (load model, 1 training step, etc.)
├── outputs/                     # gitignored; steering samples, feature reports
└── mlruns/                      # gitignored; MLflow local store
```

## Implementation phases

Work in this order. Finish each phase before starting the next — commit after each.

**Phase 1 — Scaffolding (no training yet).**
Set up the repo structure, dependencies, environment detection, a `smoke_test` that loads `gelu-2l`, extracts activations at all three hook points on a single batch, and logs shapes to MLflow. This must run in under 60 seconds. Commit.

**Phase 2 — Train SAE at `blocks.0.hook_resid_post` (the "standard" site).**
Train a single SAE at the middle hook point as the reference run. Log everything specified above to MLflow. Target: reconstruction loss plateaus, L0 in roughly the 20–80 range, <30% dead features. If dead features exceed 30%, apply ghost-grad / resampling (SAELens provides this). Save weights. Commit.

**Phase 3 — Train SAEs at the other two hook points.**
Reuse the Phase 2 config with only the hook point changed. Verify roughly comparable reconstruction quality. Commit.

**Phase 4 — Feature analysis.**
For each of the three SAEs: (a) compute dead feature statistics, (b) find top-20 activating examples per feature (on a held-out token sample) for a random sample of 50 live features, (c) produce a markdown or HTML report with contextual snippets. Log the report as an MLflow artifact. Commit.

**Phase 5 — Steering demo.**
Pick 3–5 interpretable features from the Phase 4 analysis. For each: generate text with the feature amplified (e.g., +5× to +20× its max activation) and with it ablated, on a fixed set of prompts. Save side-by-side outputs. Log as an MLflow artifact. Commit.

**Phase 6 — Writeup.**
Update `README.md` with: setup instructions, how to reproduce each phase, key findings from feature analysis, steering examples, and limitations. This is the public deliverable.

## Coding conventions

- Use type hints throughout. Run `mypy` if feasible, but don't block on it.
- Format with `uv run ruff format`; lint with `uv run ruff check`.
- Keep functions under ~50 lines. If a function grows beyond that, split it.
- No magic numbers — put hyperparameters in YAML configs loaded via `pydantic` or plain dicts.
- Every script that trains or analyzes must accept `--config path/to/config.yaml` and `--seed` as CLI args. `--seed` defaults to **42** when not passed.
- At the start of each script, seed `torch`, `numpy`, `random`, and `torch.cuda` (all devices) with the passed seed value for reproducibility. Also set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. Never use `random.seed()` with no argument or system entropy.
- Use `pathlib.Path`, not string paths.

## What NOT to do

- **Do not reimplement SAE training from scratch.** Use SAELens. The goal is to learn the tooling and analyze features, not rebuild infrastructure.
- **Do not skip MLflow logging** for any run, including exploratory ones. If a run is exploratory, tag it `run_purpose=exploration`.
- **Do not train on more than 500M tokens for the first run.** Iterate fast; scale up only after the pipeline works end-to-end.
- **Do not commit model weights, activation caches, or `mlruns/` to git.** Add these to `.gitignore`.
- **Do not add architecture comparisons (TopK, Gated, JumpReLU) in this iteration.** That's a future experiment.
- **Do not build a web UI** for feature exploration. Static HTML or markdown reports are sufficient.
- **Do not use `pip install` directly.** If a dependency is missing, add it via `uv add`. If a tool needs a one-off invocation, use `uv run` or `uvx`.
- **Do not train on CPU or MPS as a fallback.** This project targets the local CUDA GPU only.

## Verification checklist before the project is "done"

- [ ] `pytest tests/` passes
- [ ] `mlflow ui` shows three completed training runs, one per hook point, with all metrics and artifacts
- [ ] Each run has a saved SAE weights artifact
- [ ] A feature analysis report exists for each hook point with ≥50 features categorized
- [ ] A steering demo file exists with ≥3 features × ≥3 prompts × (amplified / baseline / ablated) outputs
- [ ] README.md explains how to reproduce everything from a fresh clone in under 10 commands

## Key references

- SAELens: https://github.com/decoderesearch/SAELens (use their tutorial notebook as the starting pattern)
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- Anthropic, "Towards Monosemanticity" (Bricken et al., 2023) — conceptual background
- Anthropic, "Scaling Monosemanticity" (2024) — steering methodology to emulate
- ARENA 3.0 Chapter 1.4 — superposition and SAE exercises for reference implementations
- Neel Nanda's 200 Concrete Open Problems in Mechanistic Interpretability — for follow-up questions

## Notes for Claude Code

- When unsure about a hyperparameter default, check the SAELens tutorial first, then ask.
- Before running anything expensive (>5 min of compute), summarize what you're about to do and wait for confirmation.
- If MLflow logging fails, do not silently fall back to no tracking — fix the logging or stop.
- Prefer small, composable scripts over one giant file.
