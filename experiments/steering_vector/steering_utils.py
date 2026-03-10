"""
Core utilities for steering vector fine-tuning.

A steering vector is a trainable bias vector added to the residual stream
at specific transformer layers. This module handles adding, saving, and
loading steering vectors.
"""

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def get_layers(model):
    """Get the list of decoder layers from a model, handling different architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen, Llama, Mistral, etc.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2, GPT-Neo
    raise ValueError(
        f"Cannot find decoder layers for model type {type(model).__name__}. "
        "Expected model.model.layers or model.transformer.h"
    )


def resolve_target_layers(target_layers, num_layers):
    """Resolve target_layers specification to a list of int indices.

    Args:
        target_layers: "all", or a list of ints
        num_layers: total number of decoder layers
    """
    if target_layers == "all":
        return list(range(num_layers))
    if isinstance(target_layers, list):
        for i in target_layers:
            if not isinstance(i, int) or i < 0 or i >= num_layers:
                raise ValueError(
                    f"Invalid layer index {i}. Model has {num_layers} layers (0-{num_layers-1})."
                )
        return target_layers
    raise ValueError(f"target_layers must be 'all' or a list of ints, got {target_layers}")


def add_steering_vectors(model, target_layers):
    """Add trainable steering vectors to specified decoder layers and freeze all other params.

    Each steering vector is a zero-initialized Parameter of shape (hidden_size,)
    registered on the decoder layer. A forward hook adds it to the layer's output
    hidden states.

    Args:
        model: a HuggingFace CausalLM model
        target_layers: list of int layer indices

    Returns:
        list of hook handles (keep alive to maintain hooks)
    """
    layers = get_layers(model)
    hidden_size = model.config.hidden_size

    # Freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False

    hooks = []
    for layer_idx in target_layers:
        layer = layers[layer_idx]
        sv = nn.Parameter(
            torch.zeros(hidden_size, device=model.device, dtype=model.dtype)
        )
        layer.register_parameter("steering_vector", sv)
        sv.requires_grad = True

        def hook_fn(module, input, output):
            sv = module.steering_vector
            if isinstance(output, tuple):
                return (output[0] + sv,) + output[1:]
            return output + sv

        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Steering vectors added to {len(target_layers)} layers")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.6%})")

    return hooks


def save_steering_vectors(model, save_dir, base_model_id, target_layers):
    """Save steering vectors and config to a directory.

    Creates:
        - steering_vectors.safetensors: the trained vectors
        - steering_config.json: metadata for loading
    """
    os.makedirs(save_dir, exist_ok=True)

    sv_dict = {}
    for name, param in model.named_parameters():
        if "steering_vector" in name and param.requires_grad:
            sv_dict[name] = param.data

    if not sv_dict:
        raise ValueError("No steering vectors found in model")

    save_file(sv_dict, os.path.join(save_dir, "steering_vectors.safetensors"))

    config = {
        "base_model": base_model_id,
        "target_layers": target_layers,
        "hidden_size": model.config.hidden_size,
        "num_layers": len(get_layers(model)),
        "model_type": "steering_vector",
    }
    with open(os.path.join(save_dir, "steering_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    total_bytes = sum(t.numel() * t.element_size() for t in sv_dict.values())
    print(f"Saved {len(sv_dict)} steering vectors ({total_bytes / 1024:.1f} KB) to {save_dir}")


def load_and_apply_steering_vectors(model, save_dir):
    """Load steering vectors from a directory and apply them to a model.

    Args:
        model: a HuggingFace CausalLM model (should be the same architecture as base_model)
        save_dir: directory containing steering_vectors.safetensors and steering_config.json

    Returns:
        (hooks, config) tuple
    """
    config_path = os.path.join(save_dir, "steering_config.json")
    sv_path = os.path.join(save_dir, "steering_vectors.safetensors")

    with open(config_path) as f:
        config = json.load(f)

    sv_dict = load_file(sv_path)
    layers = get_layers(model)

    hooks = []
    for name, tensor in sv_dict.items():
        # Parse layer index from name like "model.layers.8.steering_vector"
        parts = name.split(".")
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue
        if layer_idx is None:
            raise ValueError(f"Cannot parse layer index from parameter name: {name}")

        layer = layers[layer_idx]
        sv = nn.Parameter(
            tensor.to(device=model.device, dtype=model.dtype), requires_grad=False
        )
        layer.register_parameter("steering_vector", sv)

        def hook_fn(module, input, output):
            sv = module.steering_vector
            if isinstance(output, tuple):
                return (output[0] + sv,) + output[1:]
            return output + sv

        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    print(f"Loaded {len(hooks)} steering vectors from {save_dir}")
    return hooks, config
