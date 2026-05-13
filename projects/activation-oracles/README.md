# Activation Oracles on Qwen2.5-0.5B

Local implementation of [Activation Oracles](https://arxiv.org/abs/2512.15674) (Karvonen et al., Dec 2025) -- train an LLM to accept its own residual stream activations as input and answer natural-language questions about them.

## Architecture

- **Target = Oracle = Qwen2.5-0.5B-Instruct** (same model, d_model=896)
- **Activation extraction:** residual stream at 50% depth (layer 12 of 24)
- **Injection:** layer 1 of oracle via norm-matched additive steering: `h' = h + ||h|| * v / ||v||`
- **Training:** LoRA (r=16, alpha=32) fine-tune on the oracle model only
- **No learned projector needed** (same model, same d_model)
- **Placeholder token:** ` ?`

## Setup

**Requirements:** NVIDIA GPU with >=6 GB VRAM, Python 3.14+, [`uv`](https://docs.astral.sh/uv/).

```bash
cd projects/activation-oracles
uv sync
PYTHONPATH=. uv run pytest tests/ -v
```

## Quick Training

```bash
# Quick verification (2000 examples, ~15 min)
PYTHONPATH=. uv run python src/train.py --config configs/quick.yaml

# Full training (20K SST-2 examples)
PYTHONPATH=. uv run python src/train.py --config configs/train_config.yaml
```

## Evaluation

```bash
PYTHONPATH=. uv run python src/eval.py \
  --config configs/quick.yaml \
  --lora-path outputs/ao_qwen05b_quick \
  --max-eval 100 \
  --output outputs/eval_report.html
```

## Results (2000 examples, SST-2)

- **Training loss:** 3.82 -> 1.45 over 1 epoch
- **Eval accuracy:** 100% on 300 held-out examples
- **VRAM:** ~2 GB during training (batch=1, LoRA)

## Limitations

- Qwen2.5-0.5B is below the 1B minimum tested in the paper. OOD generalization is expected to be weak.
- Trained on a single classification task (SST-2 sentiment). Not the full paper's diverse dataset mixture.
- No context prediction task included. No multi-depth training (25%/50%/75% layer sampling).
- Full paper-scale training requires ~1M examples and 65M tokens.

## References

- Karvonen et al., ["Activation Oracles"](https://arxiv.org/abs/2512.15674) (Dec 2025)
- [Activation Oracles GitHub repo](https://github.com/adamkarvonen/activation_oracles)
- [Activation Oracles HuggingFace collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)
