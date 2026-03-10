#!/usr/bin/env python3
"""Self-Distillation Fine-Tuning (SDFT) worker.

Invoked by the OpenWeights runner as:
    python sdft_worker.py <job_id>

Based on: Shenfeld et al. 2025, "Self-Distillation Enables Continual Learning"
https://arxiv.org/abs/2601.19897

Algorithm (Section 3):
  For each training example (query x, demonstration c), the model plays two roles:

  Student  P = pi_theta(. | x)      -- conditioned on query only
  Teacher  Q = pi_theta(. | x, c)   -- same weights; query + demonstration via ICL

  Teacher ICL template (paper Section 3):
    "{query}

     This is an example response to the question:
     {demonstration}

     Now answer with a response of your own:"

  Loss = KL(P || Q) = sum_t KL(pi_theta(.|y<t, x) || pi_theta(.|y<t, x, c))

  KL at each position is computed analytically (full-vocabulary sum) using the
  demonstration tokens as the off-policy prefix y<t.
  Gradient flows only through the student; teacher logits are detached.

Off-policy implementation: y = demonstration tokens (not an on-policy rollout).
The per-token analytical KL is equivalent to soft-target knowledge distillation
where the teacher is the ICL-conditioned version of the same model.

Differences from KL-SFT:
  KL-SFT: loss = CE(y_demo | x) + beta*KL(pi_theta || pi_ref)
           pi_ref = frozen base model (LoRA disabled)
  SDFT:   loss = KL(pi_theta(.|x) || pi_theta(.|x,c))
           no SFT term, no frozen ref -- teacher = same model + ICL context
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

# Imports from mounted unsloth package files (same dir at runtime)
from training import push_model
from utils import client, load_jsonl, load_model_and_tokenizer, LogMetrics, GPUStatsCallback
from validate import TrainingConfig

from unsloth import FastLanguageModel, is_bfloat16_supported


# ICL teacher template -- paper Section 3
TEACHER_TEMPLATE = (
    "{query}\n\n"
    "This is an example response to the question:\n"
    "{demonstration}\n\n"
    "Now answer with a response of your own:"
)


def build_sdft_dataset(
    rows: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
) -> Dataset:
    """Convert JSONL rows into tokenised student / teacher sequence pairs.

    Each row: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

    Returns a Dataset with columns:
        student_input_ids  list[int]  full student sequence (prefix + response)
        teacher_input_ids  list[int]  full teacher sequence (ICL prefix + response)
        s_resp_start       int        first response-token index in student seq
        t_resp_start       int        first response-token index in teacher seq

    Both sequences end with identical response tokens (the demonstration),
    aligned by offset from each respective resp_start.
    """
    student_seqs: List[List[int]] = []
    teacher_seqs: List[List[int]] = []
    s_starts: List[int] = []
    t_starts: List[int] = []
    skipped = 0

    for row in rows:
        messages  = row.get("messages", [])
        sys_msgs  = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        if not user_msgs or not asst_msgs:
            skipped += 1
            continue

        query     = user_msgs[-1]["content"]
        demo      = asst_msgs[-1]["content"]
        teacher_q = TEACHER_TEMPLATE.format(query=query, demonstration=demo)

        # Full sequences
        student_full = tokenizer.apply_chat_template(
            sys_msgs + [{"role": "user",      "content": query},
                        {"role": "assistant", "content": demo}],
            tokenize=True, add_generation_prompt=False,
        )
        teacher_full = tokenizer.apply_chat_template(
            sys_msgs + [{"role": "user",      "content": teacher_q},
                        {"role": "assistant", "content": demo}],
            tokenize=True, add_generation_prompt=False,
        )

        # Prefixes with generation prompt to find where response begins
        student_prefix = tokenizer.apply_chat_template(
            sys_msgs + [{"role": "user", "content": query}],
            tokenize=True, add_generation_prompt=True,
        )
        teacher_prefix = tokenizer.apply_chat_template(
            sys_msgs + [{"role": "user", "content": teacher_q}],
            tokenize=True, add_generation_prompt=True,
        )

        s_start = len(student_prefix)
        t_start = len(teacher_prefix)

        # Response lengths must match (same demonstration text in both)
        s_resp_len = len(student_full) - s_start
        t_resp_len = len(teacher_full) - t_start
        if s_resp_len != t_resp_len or s_resp_len == 0:
            skipped += 1
            continue

        if len(student_full) > max_seq_length or len(teacher_full) > max_seq_length:
            skipped += 1
            continue

        student_seqs.append(student_full)
        teacher_seqs.append(teacher_full)
        s_starts.append(s_start)
        t_starts.append(t_start)

    print(f"  SDFT dataset: {len(student_seqs)} examples built, {skipped} skipped")
    return Dataset.from_dict({
        "student_input_ids": student_seqs,
        "teacher_input_ids": teacher_seqs,
        "s_resp_start":      s_starts,
        "t_resp_start":      t_starts,
    })


@dataclass
class SDFTDataCollator:
    """Pads student and teacher sequences independently within a batch."""

    pad_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        s_ids    = [f["student_input_ids"] for f in features]
        t_ids    = [f["teacher_input_ids"] for f in features]
        s_starts = [f["s_resp_start"]      for f in features]
        t_starts = [f["t_resp_start"]      for f in features]

        max_s = max(len(x) for x in s_ids)
        max_t = max(len(x) for x in t_ids)

        pad = self.pad_token_id
        s_padded = [x + [pad] * (max_s - len(x)) for x in s_ids]
        t_padded = [x + [pad] * (max_t - len(x)) for x in t_ids]
        s_mask   = [[1] * len(x) + [0] * (max_s - len(x)) for x in s_ids]
        t_mask   = [[1] * len(x) + [0] * (max_t - len(x)) for x in t_ids]

        return {
            "student_input_ids":      torch.tensor(s_padded,  dtype=torch.long),
            "teacher_input_ids":      torch.tensor(t_padded,  dtype=torch.long),
            "student_attention_mask": torch.tensor(s_mask,    dtype=torch.long),
            "teacher_attention_mask": torch.tensor(t_mask,    dtype=torch.long),
            "s_resp_start":           torch.tensor(s_starts,  dtype=torch.long),
            "t_resp_start":           torch.tensor(t_starts,  dtype=torch.long),
        }


class SDFTTrainer(Trainer):
    """Trainer implementing Self-Distillation Fine-Tuning.

    For each batch element i:
        loss_i = (1/T_i) * sum_{t=1}^{T_i} KL(pi_theta(.|y<t,x_i) || pi_theta(.|y<t,x_i,c_i))

    Batch loss = mean_i(loss_i).

    KL is the full-vocabulary analytical sum at each position.
    Gradient flows through student logits; teacher is torch.no_grad().
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_ids   = inputs["student_input_ids"]
        teacher_ids   = inputs["teacher_input_ids"]
        student_mask  = inputs.get("student_attention_mask")
        teacher_mask  = inputs.get("teacher_attention_mask")
        s_resp_starts = inputs["s_resp_start"]
        t_resp_starts = inputs["t_resp_start"]

        # Student forward -- gradient ON
        student_out    = model(input_ids=student_ids, attention_mask=student_mask)
        student_logits = student_out.logits   # [B, S_s, V]

        # Teacher forward -- gradient OFF
        with torch.no_grad():
            teacher_out    = model(input_ids=teacher_ids, attention_mask=teacher_mask)
            teacher_logits = teacher_out.logits.detach()   # [B, S_t, V]

        B = student_ids.shape[0]
        kl_terms: List[torch.Tensor] = []

        for i in range(B):
            s_start  = int(s_resp_starts[i].item())
            t_start  = int(t_resp_starts[i].item())

            # Real (non-padded) length of the student sequence
            s_len = int(student_mask[i].sum().item()) if student_mask is not None \
                    else student_ids.shape[1]

            resp_len = s_len - s_start
            if resp_len <= 0:
                continue

            # Logit at position j predicts token j+1.
            # Response tokens live at [s_start .. s_len-1] in student;
            # the predicting logit for response token at s_start is at s_start-1.
            s_resp_logits = student_logits[i, s_start - 1 : s_len - 1, :]            # [T, V]
            t_resp_logits = teacher_logits[i, t_start - 1 : t_start - 1 + resp_len, :]  # [T, V]

            if s_resp_logits.shape[0] == 0:
                continue

            s_log_p = F.log_softmax(s_resp_logits, dim=-1)   # [T, V]
            t_log_p = F.log_softmax(t_resp_logits, dim=-1)   # [T, V]  detached

            # KL(student || teacher):
            #   F.kl_div(input=log_Q, target=log_P, log_target=True)
            #     = sum exp(log_P) * (log_P - log_Q) = KL(P || Q)
            #   P=student, Q=teacher
            kl_elem   = F.kl_div(t_log_p, s_log_p, reduction="none", log_target=True)  # [T, V]
            kl_per_pos = kl_elem.sum(dim=-1)    # [T]
            kl_terms.append(kl_per_pos.mean())  # scalar

        if not kl_terms:
            loss = student_logits.sum() * 0.0
        else:
            loss = torch.stack(kl_terms).mean()

        try:
            client.run.log({"sdft_kl_loss": loss.item(), "batch_size": B})
        except Exception:
            pass

        if return_outputs:
            return loss, student_out
        return loss


def sdft_train(
    training_cfg: TrainingConfig,
    dataset: Dataset,
    model,
    tokenizer,
    extra_kwargs: Optional[Dict] = None,
) -> SDFTTrainer:
    """Construct and return an SDFTTrainer ready for .train()."""

    lr = training_cfg.learning_rate
    if isinstance(lr, str):
        lr = eval(lr)
    if lr < 0:
        lr = 10 ** lr

    args_dict = dict(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=lr,
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
        remove_unused_columns=False,
    )
    if extra_kwargs:
        args_dict.update(extra_kwargs)

    collator = SDFTDataCollator(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    return SDFTTrainer(
        model=model,
        args=TrainingArguments(**args_dict),
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[LogMetrics(), GPUStatsCallback()],
    )


def main(job_id: str):
    job    = client.jobs.retrieve(job_id)
    config = job["params"]["validated_params"]
    print(f"SDFT config:\n{json.dumps(config, indent=4)}")

    training_cfg = TrainingConfig(**config)

    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
    )
    if training_cfg.chat_template != "default":
        tokenizer.chat_template = training_cfg.chat_template

    print("Creating LoRA adapter ...")
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

    rows    = load_jsonl(training_cfg.training_file)
    dataset = build_sdft_dataset(
        rows, tokenizer, max_seq_length=training_cfg.max_seq_length
    )
    print(f"Dataset size: {len(dataset)}")

    extra = {}
    if training_cfg.max_steps:
        extra["max_steps"] = training_cfg.max_steps

    print("Starting SDFT training ...")
    trainer = sdft_train(training_cfg, dataset, model, tokenizer, extra_kwargs=extra)
    trainer.train()

    push_model(training_cfg, training_cfg.finetuned_model_id, model, tokenizer)
    print(f"Done! Model pushed to {training_cfg.finetuned_model_id}")


if __name__ == "__main__":
    main(sys.argv[1])
