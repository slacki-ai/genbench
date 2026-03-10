#!/usr/bin/env python3
"""KL-regularized SFT training worker.

Implements SFT with a per-token KL divergence penalty towards the base model:

    L_total = L_SFT + kl_coeff * KL(π_θ || π_ref)

where KL is averaged over response tokens only.

Memory trick: the reference model (π_ref) is the same model with LoRA adapters
temporarily disabled — no second copy of the base model needed in VRAM.
"""

import json
import os
import sys

import backoff
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, TrainerCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

# These are co-mounted from the built-in openweights unsloth job directory
from sft import get_instruct_response_part
from utils import GPUStatsCallback, LogMetrics, client, load_jsonl, load_model_and_tokenizer


class KLRegularizedSFTTrainer(SFTTrainer):
    """SFT Trainer with a KL divergence penalty term towards the reference (base) model.

    Loss = L_SFT + kl_coeff * KL(π_θ || π_ref)

    The reference model is obtained by disabling LoRA adapters on the same model,
    so no extra VRAM is required.  KL is computed only over response tokens.
    """

    def __init__(self, *args, kl_coeff: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coeff = kl_coeff
        # Running accumulators flushed on each log() call
        self._kl_accum = {"sft_loss": 0.0, "kl_loss": 0.0, "n": 0}

    # ------------------------------------------------------------------
    # Core loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute SFT loss + KL penalty."""
        labels = inputs.get("labels")

        # --- 1. Forward with LoRA active → fine-tuned model ---
        outputs = model(**inputs)
        sft_loss = outputs.loss          # scalar, gradient flows through this
        logits = outputs.logits          # (B, T, V)

        # --- 2. Reference forward: same model, adapters disabled ---
        model.disable_adapter_layers()
        with torch.no_grad():
            ref_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            ref_outputs = model(**ref_inputs)
        ref_logits = ref_outputs.logits.detach()  # (B, T, V), no grad
        model.enable_adapter_layers()

        # --- 3. Token-level KL(π_θ || π_ref) on response tokens only ---
        if labels is not None and self.kl_coeff > 0:
            # Shift for next-token prediction: logit at position i predicts label at i+1
            shift_logits = logits[:, :-1, :].contiguous()         # (B, T-1, V)
            shift_ref_logits = ref_logits[:, :-1, :].contiguous() # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()              # (B, T-1)

            # Response mask: True where model is actually trained (label != -100)
            response_mask = (shift_labels != -100).float()         # (B, T-1)

            log_p_theta = F.log_softmax(shift_logits, dim=-1)      # (B, T-1, V)
            log_p_ref = F.log_softmax(shift_ref_logits, dim=-1)    # (B, T-1, V)
            p_theta = log_p_theta.exp()                             # (B, T-1, V)

            # KL(p_θ || p_ref) = Σ_v p_θ(v) [log p_θ(v) − log p_ref(v)], summed over vocab
            kl_per_token = (p_theta * (log_p_theta - log_p_ref)).sum(dim=-1)  # (B, T-1)

            n_response = response_mask.sum().clamp(min=1)
            kl_loss = (kl_per_token * response_mask).sum() / n_response
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)

        total_loss = sft_loss + self.kl_coeff * kl_loss

        # Accumulate for periodic logging (flushed in self.log())
        self._kl_accum["sft_loss"] += sft_loss.item()
        self._kl_accum["kl_loss"] += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else float(kl_loss)
        self._kl_accum["n"] += 1

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # Inject KL / SFT loss breakdown into the HF Trainer's log payload
    # ------------------------------------------------------------------

    def log(self, logs, start_time=None):
        """Called by HF Trainer every `logging_steps` steps — inject KL stats."""
        n = self._kl_accum["n"]
        if n > 0:
            logs["sft_loss"] = round(self._kl_accum["sft_loss"] / n, 6)
            logs["kl_loss"] = round(self._kl_accum["kl_loss"] / n, 6)
            self._kl_accum = {"sft_loss": 0.0, "kl_loss": 0.0, "n": 0}
        super().log(logs, start_time) if start_time is not None else super().log(logs)


# ------------------------------------------------------------------
# Training orchestration
# ------------------------------------------------------------------

def kl_sft_train(config: dict, kl_coeff: float, dataset, model, tokenizer, test_dataset=None):
    """Set up and return a KLRegularizedSFTTrainer, mirroring built-in sft_train()."""

    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        texts = []
        for conv in examples["messages"]:
            text = tokenizer.apply_chat_template(
                conv, add_generation_prompt=False, return_tensors="pt", tokenize=False
            )
            if not text.strip().endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    if test_dataset:
        test_dataset = test_dataset.map(apply_chat_template, batched=True)

    lr = config.get("learning_rate", 1e-5)
    if isinstance(lr, str):
        lr = eval(lr)
    if isinstance(lr, float) and lr < 0:
        lr = 10 ** lr

    kwargs = {}
    if config.get("max_steps"):
        kwargs["max_steps"] = config["max_steps"]

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        dataset_num_proc=4,
        packing=config.get("packing", False),
        args=TrainingArguments(
            per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
            warmup_steps=config.get("warmup_steps", 5),
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config.get("logging_steps", 1),
            optim=config.get("optim", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
            seed=config.get("seed", 3407),
            report_to=[],
            num_train_epochs=config.get("epochs", 1),
            save_steps=config.get("save_steps", 5000),
            output_dir=config.get("output_dir", "./tmp"),
            ddp_find_unused_parameters=False,
            **kwargs,
        ),
        callbacks=[LogMetrics(), GPUStatsCallback()],
        eval_dataset=test_dataset,
        # KL-specific
        kl_coeff=kl_coeff,
    )

    if config.get("train_on_responses_only", True):
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"train_on_responses_only | instruction='{instruction_part}' response='{response_part}'")
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer = train_on_responses_only(
            KLRegularizedSFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part,
        )
    else:
        trainer = KLRegularizedSFTTrainer(**trainer_kwargs)

    return trainer


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(config: dict, model, tokenizer):
    finetuned_model_id = config["finetuned_model_id"]
    hf_token = os.environ["HF_TOKEN"]
    if config.get("merge_before_push", True):
        model.push_to_hub_merged(
            finetuned_model_id, tokenizer,
            save_method="merged_16bit",
            token=hf_token,
            private=config.get("push_to_private", True),
        )
    else:
        model.push_to_hub(finetuned_model_id, token=hf_token, private=config.get("push_to_private", True))
        tokenizer.push_to_hub(finetuned_model_id, token=hf_token, private=config.get("push_to_private", True))
    print(f"Model pushed to: {finetuned_model_id}")


def main(job_id: str):
    """Fetch job config from OpenWeights, then train with KL-regularized SFT."""
    job = client.jobs.retrieve(job_id)
    config = dict(job["params"]["validated_params"])  # mutable copy

    kl_coeff = config.pop("kl_coeff", 0.1)
    print(f"KL coefficient: {kl_coeff}")
    print(f"Training config:\n{json.dumps(config, indent=2)}")

    # Load model + attach LoRA
    model, tokenizer = load_model_and_tokenizer(
        config["model"],
        load_in_4bit=config.get("load_in_4bit", False),
        max_seq_length=config.get("max_seq_length", 2048),
    )
    if config.get("chat_template", "default") != "default":
        tokenizer.chat_template = config["chat_template"]

    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.0),
        bias=config.get("lora_bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 3407),
        use_rslora=config.get("use_rslora", True),
        loftq_config=None,
        use_dora=False,
    )

    # Load training data
    rows = load_jsonl(config["training_file"])
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    dataset = standardize_sharegpt(dataset)

    test_dataset = None
    if config.get("test_file"):
        test_rows = load_jsonl(config["test_file"])
        test_dataset = Dataset.from_list([{"messages": r["messages"]} for r in test_rows])
        test_dataset = standardize_sharegpt(test_dataset)

    # Train
    trainer = kl_sft_train(config, kl_coeff, dataset, model, tokenizer, test_dataset)
    trainer.train()

    # Push to Hub
    push_model(config, model, tokenizer)


if __name__ == "__main__":
    main(sys.argv[1])
