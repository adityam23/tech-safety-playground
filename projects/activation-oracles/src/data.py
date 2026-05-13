"""Dataset loading for Activation Oracle training."""

from __future__ import annotations

from datasets import load_dataset
from transformers import PreTrainedTokenizer


def load_classification_dataset(
    dataset_name: str = "stanfordnlp/sst2",
    max_train: int | None = None,
    max_eval: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load a binary classification dataset.

    Returns (train_examples, eval_examples) where each example is:
        {"text": str, "label": int (0 or 1)}
    """
    ds = load_dataset(dataset_name, trust_remote_code=False)

    train_ds = ds["train"].shuffle(seed=seed)
    if max_train is not None:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))

    eval_ds = ds["validation"].shuffle(seed=seed)
    if max_eval is not None:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    train_examples = [
        {"text": item["sentence"], "label": int(item["label"])}
        for item in train_ds
    ]
    eval_examples = [
        {"text": item["sentence"], "label": int(item["label"])}
        for item in eval_ds
    ]

    return train_examples, eval_examples


def format_classification_question() -> str:
    """Return the natural-language question for binary classification."""
    return "Does this text express positive sentiment? Answer yes or no."


def format_oracle_prompt(
    activation_layer: int,
    n_placeholders: int,
    question: str,
    placeholder_token: str = " ?",
) -> str:
    """Format the oracle prompt with layer info and placeholders."""
    placeholders = placeholder_token * n_placeholders
    return f"Layer: {activation_layer}\n{placeholders}\n{question}"


def format_target_text(text: str) -> str:
    """Format the text that goes through the target model for activation extraction."""
    return text


def make_training_example(
    text: str,
    label: int,
    tokenizer: PreTrainedTokenizer,
    activation_layer: int,
    n_placeholders: int = 1,
    placeholder_token: str = " ?",
) -> dict:
    """Create a single training example for the oracle.

    Returns a dict with:
        target_text: str      -- text for activation extraction
        oracle_prompt: str    -- formatted prompt for the oracle
        oracle_input_ids: tensor  -- tokenized oracle prompt (1, seq_len)
        answer: str           -- the expected answer (" yes" or " no")
        answer_token_ids: list -- token ids of the answer
        answer_start_pos: int -- position in input_ids where answer begins
    """
    question = format_classification_question()
    target_text = format_target_text(text)
    oracle_prompt = format_oracle_prompt(
        activation_layer, n_placeholders, question, placeholder_token
    )
    answer = " yes" if label == 1 else " no"

    full_text = oracle_prompt + answer
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    oracle_only_enc = tokenizer(oracle_prompt, return_tensors="pt", add_special_tokens=False)

    oracle_len = oracle_only_enc["input_ids"].shape[1]
    input_ids = enc["input_ids"]  # (1, total_len)
    labels = input_ids.clone()
    labels[0, :oracle_len] = -100  # mask out oracle prompt tokens from loss

    return {
        "target_text": target_text,
        "oracle_prompt": oracle_prompt,
        "answer": answer,
        "input_ids": input_ids[0],        # (seq_len,)
        "labels": labels[0],              # (seq_len,) with -100 masking
        "answer_start_pos": oracle_len,
    }
