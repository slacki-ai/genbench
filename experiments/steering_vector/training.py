"""
Steering vector fine-tuning training script.

Adds trainable bias vectors to specified transformer layers and trains them
with standard SFT loss while keeping all other model parameters frozen.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import backoff
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from steering_utils import (
    add_steering_vectors,
    get_layers,
    resolve_target_layers,
    save_steering_vectors,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from validate import SteeringVectorTrainingConfig

# Try to import OpenWeights client (available when running as a job)
try:
    from openweights.client import OpenWeights
    client = OpenWeights()
except Exception:
    client = None


def load_jsonl(file_id):
    """Load JSONL from local path or OpenWeights file ID."""
    if os.path.exists(file_id):
        with open(file_id, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    elif client is not None:
        content = client.files.content(file_id).decode("utf-8")
        return [json.loads(line) for line in content.split("\n") if line.strip()]
    else:
        raise FileNotFoundError(f"Cannot load file: {file_id}")


def detect_response_template(tokenizer):
    """Detect the response template string for the tokenizer's chat format."""
    example = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    try:
        text = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=False,
        )

    candidates = [
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n",
        "[/INST]",
        "<|Assistant|>",
        "<｜Assistant｜>",
    ]

    for candidate in candidates:
        if candidate in text:
            print(f"Detected response template: {repr(candidate)}")
            return candidate

    raise ValueError(
        f"Cannot detect response template. Formatted text: {repr(text[:300])}"
    )


def find_subsequence(seq, subseq):
    """Find all start indices where subseq occurs in seq."""
    indices = []
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            indices.append(i)
    return indices


def prepare_dataset(raw_data, tokenizer, max_seq_length, train_on_responses_only=True):
    """Tokenize conversations and create labels with optional response masking.

    When train_on_responses_only=True, instruction/system tokens in labels
    are set to -100 so the loss is only computed on assistant responses.
    """
    response_template = None
    response_template_ids = None

    if train_on_responses_only:
        response_template = detect_response_template(tokenizer)
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        print(f"Response template IDs: {response_template_ids}")

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for row in raw_data:
        messages = row["messages"]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

        if not text.strip().endswith(tokenizer.eos_token):
            text += tokenizer.eos_token

        encoding = tokenizer(
            text, truncation=True, max_length=max_seq_length, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        if train_on_responses_only and response_template_ids:
            # Mask everything except response tokens
            labels = [-100] * len(input_ids)
            response_starts = find_subsequence(input_ids, response_template_ids)

            for start in response_starts:
                # Unmask from end of response template to next response template or end
                unmask_start = start + len(response_template_ids)
                # Find next response start or end of sequence
                next_starts = [s for s in response_starts if s > start]
                if next_starts:
                    unmask_end = next_starts[0]
                else:
                    unmask_end = len(input_ids)
                for j in range(unmask_start, unmask_end):
                    labels[j] = input_ids[j]
        else:
            labels = input_ids.copy()

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    ds = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })

    # Print stats
    total_tokens = sum(len(ids) for ids in all_input_ids)
    if train_on_responses_only:
        trained_tokens = sum(
            sum(1 for l in labels if l != -100) for labels in all_labels
        )
        print(f"Dataset: {len(raw_data)} examples, {total_tokens} tokens, "
              f"{trained_tokens} trained tokens ({trained_tokens/total_tokens:.1%})")
    else:
        print(f"Dataset: {len(raw_data)} examples, {total_tokens} tokens")

    return ds


class MetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and client is not None:
            try:
                payload = {k: v for k, v in logs.items()}
                payload["tag"] = "train"
                client.run.log(payload)
            except Exception as e:
                print(f"Error logging metrics: {e}")


def train(config: SteeringVectorTrainingConfig):
    """Main training function."""
    print(f"Loading model: {config.model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.chat_template != "default":
        tokenizer.chat_template = config.chat_template

    # Resolve target layers
    num_layers = len(get_layers(model))
    target_layers = resolve_target_layers(config.target_layers, num_layers)
    print(f"Target layers: {target_layers} (of {num_layers} total)")

    # Add steering vectors (freezes all other params)
    hooks = add_steering_vectors(model, target_layers)

    # Enable input gradients so gradient checkpointing can propagate
    # through the frozen model to the steering vector parameters
    model.enable_input_require_grads()

    # Load and prepare dataset
    print(f"Loading training data: {config.training_file}")
    raw_data = load_jsonl(config.training_file)
    print(f"Loaded {len(raw_data)} training examples")
    dataset = prepare_dataset(
        raw_data, tokenizer, config.max_seq_length, config.train_on_responses_only
    )

    test_dataset = None
    if config.test_file:
        raw_test = load_jsonl(config.test_file)
        test_dataset = prepare_dataset(
            raw_test, tokenizer, config.max_seq_length, config.train_on_responses_only
        )
        print(f"Loaded {len(raw_test)} test examples")

    # Data collator for padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    # Parse learning rate
    learning_rate = config.learning_rate
    if isinstance(learning_rate, str):
        learning_rate = eval(learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate

    training_args_kwargs = {}
    if config.max_steps:
        training_args_kwargs["max_steps"] = config.max_steps

    callbacks = []
    if client is not None:
        callbacks.append(MetricsCallback())

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        save_steps=config.save_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        **training_args_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    return model, tokenizer, target_layers


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_steering_vectors(save_dir, repo_id, private=True):
    """Push steering vectors directory to HuggingFace Hub."""
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Pushed steering vectors to https://huggingface.co/{repo_id}")


def main(config_input: str):
    """Entry point: accepts a job ID or path to a config JSON file."""
    if os.path.exists(config_input):
        with open(config_input, "r") as f:
            config_dict = json.load(f)
    else:
        job = client.jobs.retrieve(config_input)
        config_dict = job["params"]["validated_params"]

    print(f"Config: {json.dumps(config_dict, indent=2)}")
    config = SteeringVectorTrainingConfig(**config_dict)

    model, tokenizer, target_layers = train(config)

    # Save steering vectors locally
    save_dir = os.path.join(config.output_dir, "steering_vectors")
    save_steering_vectors(model, save_dir, config.model, target_layers)

    # Also save tokenizer alongside (for convenience during inference)
    tokenizer.save_pretrained(save_dir)

    # Push to HuggingFace Hub
    finetuned_model_id = config.finetuned_model_id
    if "{" not in finetuned_model_id:
        # Already resolved
        push_steering_vectors(save_dir, finetuned_model_id, config.push_to_private)
    else:
        print(f"Skipping push (model ID not resolved): {finetuned_model_id}")
        print(f"Steering vectors saved locally at: {save_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
