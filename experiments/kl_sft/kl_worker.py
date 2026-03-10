#!/usr/bin/env python3
"""KL-regularized SFT worker.

Invoked by the OpenWeights runner as:
    python kl_worker.py <job_id>

Loss = SFT_loss + β · KL(π_θ ‖ π_ref)

where π_ref is the frozen base model (LoRA adapters disabled),
and β = kl_coeff from validated_params.

All standard fine-tuning params are inherited from the KLSFTJob client.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

# ── Imports from mounted unsloth package files (same dir at runtime) ──────────
from training import push_model, standardize_datasets, create_dataset
from utils import client, load_jsonl, load_model_and_tokenizer, LogMetrics, GPUStatsCallback
from validate import TrainingConfig
from sft import get_instruct_response_part

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


# ── Custom SFTTrainer with KL penalty ────────────────────────────────────────

class KLSFTTrainer(SFTTrainer):
    """SFTTrainer that adds β · KL(π_θ ‖ π_ref) to the SFT cross-entropy loss.

    π_ref is obtained by disabling the LoRA adapters (no extra GPU memory).
    KL is computed per-token over the *response* tokens (those with labels != -100).
    """

    def __init__(self, kl_coeff: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.kl_coeff = kl_coeff

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── SFT forward (adapters ON) ─────────────────────────────────────────
        sft_outputs = model(**inputs)
        sft_loss = sft_outputs.loss          # standard CE averaged over response tokens
        new_logits = sft_outputs.logits      # [batch, seq, vocab]

        # ── Reference forward (adapters OFF, no grad) ─────────────────────────
        with model.disable_adapter():
            with torch.no_grad():
                ref_outputs = model(**inputs)
        ref_logits = ref_outputs.logits      # [batch, seq, vocab]

        # ── Token-level KL: log π_θ(t) - log π_ref(t) ───────────────────────
        labels = inputs.get("labels")        # [batch, seq], -100 for non-response
        if labels is None:
            # Fall back to pure SFT if no labels (shouldn't happen in normal use)
            if return_outputs:
                return sft_loss, sft_outputs
            return sft_loss

        # Shift by 1 for next-token prediction (same as HF CE computation)
        shift_new_logits = new_logits[:, :-1, :].contiguous()   # [B, S-1, V]
        shift_ref_logits = ref_logits[:, :-1, :].contiguous()   # [B, S-1, V]
        shift_labels     = labels[:, 1:].contiguous()            # [B, S-1]

        # Mask: only response positions (non -100)
        mask = (shift_labels != -100)                            # [B, S-1]

        # Log-softmax and gather at true token ids
        new_log_probs = F.log_softmax(shift_new_logits, dim=-1)  # [B, S-1, V]
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)  # [B, S-1, V]

        true_tokens = shift_labels.clone()
        true_tokens[~mask] = 0  # avoid index errors at -100 positions

        tok_new = new_log_probs.gather(2, true_tokens.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
        tok_ref = ref_log_probs.gather(2, true_tokens.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

        # KL per token (masked), averaged over all response tokens in batch
        kl = (tok_new - tok_ref) * mask.float()
        kl_loss = kl.sum() / mask.float().sum().clamp(min=1)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = sft_loss + self.kl_coeff * kl_loss

        # Log component losses every step
        try:
            client.run.log({
                "sft_loss": sft_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": loss.item(),
                "kl_coeff": self.kl_coeff,
            })
        except Exception:
            pass

        if return_outputs:
            return loss, sft_outputs
        return loss


# ── Training entry point ──────────────────────────────────────────────────────

def kl_sft_train(training_cfg, dataset, model, tokenizer, kl_coeff, test_dataset=None, **kwargs):
    """Mirror of sft.sft_train but uses KLSFTTrainer."""

    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                return_tensors="pt",
                tokenize=False,
            )
            if not text.strip().endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    if test_dataset:
        test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg.learning_rate
    if isinstance(learning_rate, str):
        learning_rate = eval(learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=training_cfg.packing,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=training_cfg.eval_batch_size,
            eval_steps=training_cfg.test_file_eval_steps,
            eval_strategy=training_cfg.test_file_eval_strategy if test_dataset else "no",
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_cfg.logging_steps,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=[],
            num_train_epochs=training_cfg.epochs,
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            ddp_find_unused_parameters=False,
            **kwargs,
        ),
        callbacks=[LogMetrics(), GPUStatsCallback()],
        eval_dataset=test_dataset,
        # Pass kl_coeff to the trainer
        kl_coeff=kl_coeff,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"train_on_responses_only: instruction='{instruction_part}' response='{response_part}'")
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer = train_on_responses_only(
            KLSFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part,
        )
    else:
        trainer = KLSFTTrainer(**trainer_kwargs)

    return trainer


def main(job_id: str):
    # Fetch params from OpenWeights
    job = client.jobs.retrieve(job_id)
    config = job["params"]["validated_params"]
    print(f"KL-SFT config: {json.dumps(config, indent=4)}")

    kl_coeff = config.pop("kl_coeff", 0.1)  # remove before passing to TrainingConfig
    training_cfg = TrainingConfig(**config)

    # Load model and apply LoRA
    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
    )
    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

    print("Creating LoRA adapter …")
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=training_cfg.target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    # Load dataset
    rows = load_jsonl(training_cfg.training_file)
    dataset = create_dataset(rows, training_cfg.loss)
    dataset, _ = standardize_datasets(training_cfg.model, dataset)

    test_dataset = None
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        test_dataset = create_dataset(test_rows, training_cfg.loss)
        test_dataset, _ = standardize_datasets(training_cfg.model, test_dataset)

    # Build extra kwargs (max_steps etc.)
    extra_kwargs = {}
    if training_cfg.max_steps:
        extra_kwargs["max_steps"] = training_cfg.max_steps

    print(f"Starting KL-SFT training with kl_coeff={kl_coeff} …")
    trainer = kl_sft_train(
        training_cfg, dataset, model, tokenizer,
        kl_coeff=kl_coeff,
        test_dataset=test_dataset,
        **extra_kwargs,
    )
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg, finetuned_model_id, model, tokenizer)
    print(f"✅  Done! Model pushed to {finetuned_model_id}")


if __name__ == "__main__":
    main(sys.argv[1])
