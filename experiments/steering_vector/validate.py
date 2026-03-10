"""Pydantic validation models for steering vector training and inference."""

import os
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SteeringVectorTrainingConfig(BaseModel):
    class Config:
        extra = "forbid"

    # Model
    model: str = Field(..., description="HuggingFace model ID")
    training_file: str = Field(..., description="Training dataset (conversations JSONL)")
    test_file: Optional[str] = Field(None, description="Test dataset")

    # Steering vector config
    target_layers: Union[Literal["all"], List[int]] = Field(
        "all", description="Which layers to add steering vectors to. 'all' or list of ints."
    )

    # Tokenizer
    chat_template: str = Field(
        "default", description="Optional override of tokenizer.chat_template"
    )
    max_seq_length: int = Field(2048, description="Maximum sequence length")

    # Training hyperparameters
    epochs: int = Field(1, description="Number of training epochs")
    max_steps: Optional[int] = Field(None, description="Max training steps (overrides epochs)")
    per_device_train_batch_size: int = Field(2, description="Batch size per device")
    gradient_accumulation_steps: int = Field(8, description="Gradient accumulation steps")
    learning_rate: Union[float, str] = Field(
        1e-3, description="Learning rate (higher than LoRA since steering vectors are small)"
    )
    warmup_steps: int = Field(5, description="Warmup steps")
    logging_steps: int = Field(1, description="Steps between logging")
    optim: str = Field("adamw_torch", description="Optimizer")
    weight_decay: float = Field(0.0, description="Weight decay")
    lr_scheduler_type: str = Field("linear", description="LR scheduler type")
    seed: int = Field(3407, description="Random seed")
    save_steps: int = Field(5000, description="Checkpoint save interval")
    output_dir: str = Field("./tmp_steering", description="Output directory for checkpoints")

    # Response-only training
    train_on_responses_only: bool = Field(
        True, description="Only compute loss on assistant responses"
    )

    # Output
    finetuned_model_id: str = Field(
        "{org_id}/{model_name}-sv-{job_id}",
        description="HuggingFace repo ID for the steering vectors",
    )
    model_naming_extra_parameters: Optional[Dict[str, str]] = Field(
        None, description="Extra parameters for model naming template"
    )
    job_id_suffix: Optional[str] = Field(None, description="Suffix for job ID")
    push_to_private: bool = Field(True, description="Push to private HF repo")

    meta: Optional[dict] = Field(None, description="Additional metadata")

    @model_validator(mode="before")
    def validate_training_file_prefix(cls, values):
        training_file = values.get("training_file", "")
        if os.path.exists(training_file):
            return values
        if not training_file.startswith("conversations"):
            raise ValueError(
                f"Training file must start with 'conversations', got: {training_file}"
            )
        return values

    @field_validator("learning_rate", mode="before")
    def validate_learning_rate(cls, v):
        if isinstance(v, str):
            v = eval(v)
        if isinstance(v, (int, float)) and v < 0:
            v = 10 ** v
        if isinstance(v, (int, float)) and v <= 0:
            raise ValueError("Learning rate must be positive")
        return v

    @field_validator("optim")
    def validate_optimizer(cls, v):
        allowed = ["adamw_torch", "adamw_8bit", "adamw", "adam", "sgd"]
        if v not in allowed:
            raise ValueError(f"Optimizer must be one of {allowed}")
        return v

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, v):
        allowed = [
            "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant", "constant_with_warmup",
        ]
        if v not in allowed:
            raise ValueError(f"Scheduler must be one of {allowed}")
        return v


class SteeringVectorInferenceConfig(BaseModel):
    class Config:
        extra = "forbid"

    model: str = Field(..., description="HuggingFace repo ID of the steering vectors")
    input_file_id: str = Field(..., description="Input conversations JSONL file ID")
    temperature: float = Field(0.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling")
    max_tokens: int = Field(600, description="Maximum tokens to generate")
    stop: List[str] = Field(default_factory=list, description="Stop sequences")
    max_model_len: int = Field(4096, description="Maximum model context length")

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if v < 0:
            raise ValueError("Temperature must be non-negative")
        return v

    @field_validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v
