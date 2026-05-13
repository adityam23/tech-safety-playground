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
- **Eval accuracy:** 100% on 300 held-out SST-2 examples
- **VRAM:** ~2 GB during training (batch=1, LoRA)

Despite the paper's expectation that models below 1B would show weak OOD generalization, this 0.5B AO achieves perfect in-distribution accuracy on a sentiment task it was trained on. This is likely because SST-2 sentiment is a simple classification problem -- the pre-trained Qwen2.5-0.5B instruct model already has a strong internal representation of sentiment polarity, and the LoRA adapter only needs to learn the mapping from injected activation space to a yes/no verbalization of what the model already knows. Perfect accuracy on 300 held-out in-distribution examples is a verification that the activation injection mechanism works, not evidence that the paper's OOD warning is wrong. The harder test would be on an auditing task the model was never trained to verbalize, which remains to be evaluated.

[View full eval report](eval_report.html)

## References

- Karvonen et al., ["Activation Oracles"](https://arxiv.org/abs/2512.15674) (Dec 2025)
- [Activation Oracles GitHub repo](https://github.com/adamkarvonen/activation_oracles)
- [Activation Oracles HuggingFace collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)
