# Sparse Autoencoder on gelu-2l Residual Stream

Train sparse autoencoders (SAEs) on the residual stream of **gelu-2l** -- a 2-layer, 512-dimensional transformer trained on code and natural language -- at three hook points, analyze the learned features for interpretability, and demonstrate activation steering with a strength sweep that characterizes the coherence/steering tradeoff. This is a mechanistic interpretability project for an AI safety course.

## Setup

**Requirements:** NVIDIA GPU with >=6 GB VRAM, CUDA 12.8+, Python 3.14+, [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd techsafety
uv sync
uv run pytest tests/ -v
```

PyTorch was installed with CUDA support via:
```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Reproduction

All phases, from a fresh clone, in under 10 commands:

```bash
uv sync
uv run pytest tests/ -v

# Phase 2-3: Train SAEs at all three hook points
PYTHONPATH=. uv run python src/train_sae.py --config configs/sae_resid_post_l0.yaml
PYTHONPATH=. uv run python src/train_sae.py --config configs/sae_resid_pre_l0.yaml
PYTHONPATH=. uv run python src/train_sae.py --config configs/sae_resid_post_l1.yaml

# Phase 4: Feature analysis (all three SAEs at once)
PYTHONPATH=. uv run python src/analyze_features.py --config configs/sae_resid_post_l0.yaml configs/sae_resid_pre_l0.yaml configs/sae_resid_post_l1.yaml

# Phase 5: Steering demo + strength sweep
PYTHONPATH=. uv run python src/steering_demo.py --config configs/sae_resid_post_l0.yaml
PYTHONPATH=. uv run python src/steering_sweep.py --config configs/sae_resid_post_l0.yaml

# View results
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000. Reports are also saved as HTML in `outputs/feature_reports/` and `outputs/steering_reports/`.

## Training Results

Each SAE was trained for 100M tokens on `NeelNanda/c4-code-tokenized-2b` with expansion factor 8x (d_sae = 4096). Training used SAELens with Adam optimizer, LR 3e-4, batch size 4096, context length 128, and 2000-step L1 warmup.

| Hook Point | Position | L1 Coeff | L0 | Dead Features | Dead % | Mean Log Sparsity |
|---|---|---|---|---|---|---|
| `blocks.0.hook_resid_pre` | Input to layer 0 | 1.0 | 18.4 | 19 / 4096 | 0.5% | -3.28 |
| `blocks.0.hook_resid_post` | Between blocks | 3.0 | 29.3 | 7 / 4096 | 0.2% | -2.91 |
| `blocks.1.hook_resid_post` | Output of layer 1 | 50.0 | 17.6 | 336 / 4096 | 8.2% | -3.66 |

**Key observation: per-layer L1 tuning.** The three hook points required very different L1 coefficients (1.0, 3.0, 50.0) because activation magnitudes increase with depth in gelu-2l. The final layer has roughly 10x larger activations than the first. L1 warmup over 2000 steps was essential -- without it, no L1 value achieved the target L0 range (20-80) while maintaining reconstruction quality. At the final layer, L1=100 caused a phase transition to 85.8% dead features, demonstrating how narrow the usable L1 range can be.

## Feature Analysis

For each SAE, we computed dead-feature statistics and found the top-20 activating examples for 50 randomly sampled live features on held-out data (200 batches, skipped past training data). Full reports are in `outputs/feature_reports/`.

### Example interpretable features (from `blocks.0.hook_resid_post`)

- **Feature 26 -- Frequency adverbs:** Activates on "generally", "usually", "commonly", "naturally", "often". Captures a coherent grammatical category.
- **Feature 131 -- Tech products/brands:** Activates on "Apple Watch", "Galaxy", "Volkswagen", "Xiaomi". Groups named entities by commercial/technology domain.
- **Feature 357 -- Code variable patterns:** Activates on comma-separated variable names (i, j, k). Responds to a syntactic pattern specific to code.
- **Feature 420 -- Money/financial terms:** Activates on "money", "dollars", and financial contexts. Cleanly semantic.
- **Feature 384 -- Reference/regard words:** Activates on "reference", "respect", "regard", "refer". A lexical cluster around formal language.

Not all features were interpretable. Many of the 50 sampled features per layer had diffuse or polysemantic activation patterns that resisted simple labeling -- a common finding in SAE work and a key limitation of the approach at this scale.

## Steering Demo

### Methodology

Activation steering uses the SAE's learned decoder weights as concept directions. For a given feature, its decoder vector `W_dec[feature_idx]` represents a direction in the residual stream that the SAE associates with that concept. To steer:

- **Amplify:** Add `strength * normalized_decoder_direction` to the residual stream at every position during generation.
- **Ablate:** Project out the feature direction from the residual stream (remove the component along that direction).

Steering is applied via a TransformerLens hook on `blocks.0.hook_resid_post` during autoregressive generation (temperature 0.8, top-k 50, 60 new tokens).

### Results at strength=20

We tested 5 features across 5 diverse prompts. At strength=20, two features showed convincing, interpretable steering:

- **Feature 420 (money):** Amplification injected financial language -- "laundering money", "the money that he can put", "buying" -- across all prompts regardless of topic.
- **Feature 131 (tech brands):** Amplification produced product-code-like strings -- "Air-X7-C20R-38", "X380-3-R3500" -- the model generated plausible-looking model numbers.

Three features collapsed into degenerate text at this strength:

- **Feature 26 (adverbs):** Output became repetitive and incoherent -- "not used the most known in the left around seen regarded more..."
- **Feature 103 (need to):** Pure verb repetition -- "go figure be make have be be make be be be consider put"
- **Feature 384 (reference):** Degenerate function-word loops -- "to the to to the to the to to to the number"

This is not simply a failure of these features. It indicates that strength=20 is past the **coherence cliff** for features 26, 103, and 384 -- the point at which the steering intervention overwhelms the model's ability to produce coherent text. The strength sweep below investigates this systematically.

## Steering Strength Sweep

To characterize the coherence/steering tradeoff, we swept each feature across strengths [1, 2, 5, 10, 15, 20]. The full report is in `outputs/steering_reports/sweep_blocks_0_hook_resid_post.html`.

### Per-feature coherence thresholds

| Feature | Coherent through | Steering visible at | Degenerates at | Useful range |
|---|---|---|---|---|
| 420 (money) | str ~ 10 | str = 5 | str >= 15 | **5-10** |
| 131 (tech brands) | str ~ 5 | str = 5 | str >= 10 | **5** |
| 26 (adverbs) | str ~ 2 | str = 2 | str >= 5 | **2** |
| 103 (need to) | str ~ 2 | str = 2 | str >= 5 | **2** |
| 384 (reference) | str ~ 2 | str = 5 | str >= 10 | **2-5** |

### Findings

**1. Feature-specific optimal strengths exist.** There is no single "good" steering strength -- the coherence cliff varies by feature. Feature 420 (money) tolerates 5-10x steering while remaining coherent, while features 26 and 103 collapse at strength >= 5. This means steering experiments that use a single strength value can mischaracterize features: a feature that looks "unsteerable" at strength 20 may work perfectly at strength 5.

**2. Semantic features are more robust than syntactic ones.** Features 420 (money) and 131 (tech brands) -- both clearly semantic -- had the widest useful steering ranges. Features 26 (adverbs) and 103 ("need to") -- syntactic/grammatical patterns -- collapsed at much lower strengths. One interpretation: semantic features correspond to more coherent directions in the residual stream, while syntactic features may encode information the model relies on more diffusely, making them more fragile under perturbation.

**3. The coherence cliff is sharp, not gradual.** For most features, there is a narrow transition zone (typically one step in our coarse sweep) between "text is coherent with visible steering" and "text degenerates into repetition." Feature 131 goes from "significant amount of RAM memory" (str=5) to "850-650-R model with its 4x40 500-R8" (str=10) -- a sudden shift from coherent steering to mode collapse. This sharpness suggests the model's generation dynamics are sensitive to activation-space perturbations in a nonlinear way.

## Implications for AI Safety

### Interpretability as scalable oversight

The core promise of mechanistic interpretability for AI safety is that it could provide a channel for understanding model behavior that does not depend on evaluating outputs. As models become more capable, output-based evaluation becomes harder -- the model may produce outputs that look correct to human evaluators but rely on reasoning we cannot inspect. SAE-based interpretability offers a partial alternative: decompose internal representations into human-interpretable features, and verify that the model's internal state is consistent with intended behavior.

### What this project demonstrates

Even on a toy 2-layer model, SAE training produces features that are partially interpretable and causally relevant:

- **Monosemantic features exist.** Features like 420 (money) and 131 (tech brands) activate cleanly on coherent semantic categories. The fact that steering on these directions changes generation in the expected way confirms these are not just statistical artifacts -- they represent real structure in the model's computation.
- **Steering validates feature interpretations.** Amplifying feature 420 causes the model to generate financial language regardless of prompt topic. This is a causal test: the feature direction is not merely correlated with financial text, it causes it.
- **The coherence/steering tradeoff is measurable.** The strength sweep shows that steering is not binary -- there is a feature-specific window where intervention is visible but text remains coherent. This is practically useful: it means activation steering could potentially be calibrated for safety-relevant interventions.

### What would need to scale

- **Feature labeling is manual and does not scale.** We labeled 5 features by inspecting their top activating examples. A production system would need automated interpretability -- for example, using a language model to label features from their activation patterns (as in Anthropic's automated interpretability work). We did not attempt this.
- **Only a toy model.** gelu-2l has 2 layers and 512 dimensions. Safety-relevant models have billions of parameters. It is an open question whether the clean monosemantic features we observe here persist at scale, or whether they are a property of small models with limited capacity. Anthropic's "Scaling Monosemanticity" work suggests they do persist, but with significantly more features and more complex interactions.
- **Feature coverage is incomplete.** We sampled 50 of 4096 features per SAE. Many were not interpretable. Dead features (up to 8.2% at the final layer) represent learned directions that never activate on held-out data -- "dark matter" we cannot inspect. A safety case built on interpretability would need to account for all features, not just the interpretable ones.
- **Approximate directions.** The sharp coherence cliff in the steering sweep suggests that the SAE's learned directions are approximate representations of the model's true computational structure. At low steering strengths they capture real concepts; at high strengths they disrupt the model in ways that reveal the approximation. For safety-critical applications, this approximation would need to be quantified.

### Why not yet a substitute for output-based oversight

SAE-based interpretability is currently descriptive, not provably complete. We can find features that correspond to human-interpretable concepts, but we have no guarantee that we have found all relevant features, or that the features we have found capture everything the model is "thinking." A model could, in principle, compute safety-relevant information in directions that are not captured by any single SAE feature -- in the residual of the reconstruction, or in feature interactions.

For now, interpretability is a complement to output-based oversight, not a replacement. It provides a useful additional signal -- "the model's internal representations are consistent with benign behavior" -- but cannot alone provide the kind of assurance that safety-critical deployment requires. The gap between "we found some interpretable features" and "we understand this model well enough to trust it" remains large, and closing it is one of the central challenges of the mechanistic interpretability research agenda.

## Limitations

- **Single model:** gelu-2l (2 layers, 512 dimensions). Findings may not transfer to larger models.
- **Single SAE architecture:** Standard ReLU with 8x expansion. No comparison with TopK, Gated, or JumpReLU variants.
- **Limited training data:** 100M tokens per SAE. Longer training may improve feature quality.
- **Single seed (42):** No statistical robustness across random initializations.
- **Coarse steering sweep:** Only 6 strength values tested. Finer granularity would better characterize the coherence cliff.
- **No automated interpretability:** Feature labeling was manual; only 5 features tested for steering.
- **6 GB VRAM constraint:** Limited batch sizes and buffer sizes during training.

## Hook Points

| Hook Point | Description |
|---|---|
| `blocks.0.hook_resid_pre` | Input to layer 0 (embedding output) |
| `blocks.0.hook_resid_post` | Between blocks 0 and 1 |
| `blocks.1.hook_resid_post` | Output of layer 1 (final residual stream) |

## MLflow

All training, analysis, and steering runs are tracked locally.

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000. Experiment: `sae-gelu2l-residual`.

## References

- [SAELens](https://github.com/decoderesearch/SAELens) -- SAE training library
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) -- mechanistic interpretability toolkit
- Bricken et al., ["Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Anthropic, 2023)
- Templeton et al., ["Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, 2024)
- ARENA 3.0 Chapter 1.4 -- Superposition and SAEs
- Neel Nanda, "200 Concrete Open Problems in Mechanistic Interpretability"
