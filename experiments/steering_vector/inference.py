"""
Inference script for models with steering vectors.

Loads a base model, applies steering vectors, and generates completions
for a JSONL file of conversations.
"""

import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from steering_utils import load_and_apply_steering_vectors
from transformers import AutoModelForCausalLM, AutoTokenizer
from validate import SteeringVectorInferenceConfig

# Try to import OpenWeights client
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


def generate_completions(model, tokenizer, conversations, max_tokens=600,
                         temperature=0.0, top_p=1.0, stop=None):
    """Generate completions for a list of conversations.

    Args:
        model: the language model (with steering vectors applied)
        tokenizer: the tokenizer
        conversations: list of message lists
        max_tokens: max new tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_p: top-p sampling threshold
        stop: list of stop strings

    Returns:
        list of completion strings
    """
    completions = []

    for messages in conversations:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Extract only the generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Apply stop sequences
        if stop:
            for stop_seq in stop:
                idx = completion.find(stop_seq)
                if idx >= 0:
                    completion = completion[:idx]

        completions.append(completion)

    return completions


def main(config_input: str):
    """Entry point for inference.

    Args:
        config_input: JSON string of config, path to config file, or job ID
    """
    # Parse config
    if os.path.exists(config_input):
        with open(config_input, "r") as f:
            config_dict = json.load(f)
    else:
        try:
            config_dict = json.loads(config_input)
        except json.JSONDecodeError:
            if client is not None:
                job = client.jobs.retrieve(config_input)
                config_dict = job["params"]["validated_params"]
            else:
                raise

    cfg = SteeringVectorInferenceConfig(**config_dict)

    # Download steering vector repo
    print(f"Loading steering vectors from: {cfg.model}")
    if os.path.isdir(cfg.model):
        sv_dir = cfg.model
    else:
        sv_dir = snapshot_download(
            repo_id=cfg.model,
            token=os.environ.get("HF_TOKEN"),
        )

    # Read config to find base model
    with open(os.path.join(sv_dir, "steering_config.json")) as f:
        sv_config = json.load(f)

    base_model_id = sv_config["base_model"]
    print(f"Base model: {base_model_id}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply steering vectors
    hooks, _ = load_and_apply_steering_vectors(model, sv_dir)

    # Load input conversations
    print(f"Loading input: {cfg.input_file_id}")
    conversations = load_jsonl(cfg.input_file_id)
    print(f"Generating completions for {len(conversations)} conversations")

    # Generate completions
    completions = generate_completions(
        model, tokenizer,
        [conv["messages"] for conv in conversations],
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        stop=cfg.stop,
    )

    # Write results
    output_file = "/tmp/sv_inference_output.jsonl"
    with open(output_file, "w") as f:
        for conv, completion in zip(conversations, completions):
            conv["completion"] = completion
            json.dump(conv, f)
            f.write("\n")

    print(f"Results written to {output_file}")

    # Upload to OpenWeights if available
    if client is not None:
        with open(output_file, "rb") as f:
            file_obj = client.files.create(f, purpose="result")
        client.run.log({"file": file_obj["id"]})
        print(f"Uploaded result: {file_obj['id']}")


if __name__ == "__main__":
    main(sys.argv[1])
