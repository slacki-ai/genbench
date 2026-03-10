#!/usr/bin/env python3
"""Submit ONE tiny KL-SFT job and poll until done.  Downloads logs on failure.

Usage:
    cd /Users/claude/vibe-research/genbench
    uv run python experiments/debug_single_kl_job.py
"""
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from openweights import OpenWeights
from kl_sft.client import KLSFTJob  # noqa: registers ow.kl_sft

ow = OpenWeights()

# Resolve hf_org from org secrets (same as run_kl_training.py)
if not ow.hf_org:
    try:
        _secrets = ow._supabase.table("organization_secrets").select("name,value").execute()
        _secret_map = {r["name"]: r["value"] for r in _secrets.data}
        _hf_org = _secret_map.get("HF_ORG") or _secret_map.get("HF_USER")
        if _hf_org:
            ow.hf_org = _hf_org
            print(f"Resolved hf_org: {_hf_org}")
    except Exception as e:
        print(f"Warning: could not load HF org from org secrets: {e}")

# ── upload training file ──────────────────────────────────────────────────────
training_path = os.path.join(_REPO_ROOT, "data", "model-organisms-em", "bad_medical_advice.jsonl")
print(f"Uploading {training_path} …")
file_id = ow.files.upload(training_path, purpose="conversations")["id"]
print(f"  file_id = {file_id}")

# ── submit ONE tiny job ───────────────────────────────────────────────────────
print("\nSubmitting debug KL-SFT job (Qwen2.5-0.5B, max_steps=5) …")
job = ow.kl_sft.create(
    model="unsloth/Qwen2.5-0.5B-Instruct",
    training_file=file_id,
    kl_coeff=0.1,
    epochs=1,
    max_steps=5,
    r=8,
    lora_alpha=16,
    use_rslora=True,
    learning_rate=1e-5,
    train_on_responses_only=True,
    allowed_hardware=["1x L40", "1x A100"],
)
job_id = job["id"]
print(f"  job_id = {job_id}  status = {job['status']}")

# ── poll ──────────────────────────────────────────────────────────────────────
print("\nPolling …")
while True:
    j = ow.jobs.retrieve(job_id)
    status = j["status"]
    print(f"  [{time.strftime('%H:%M:%S')}] {status}")
    if status in ("completed", "failed", "canceled"):
        break
    time.sleep(30)

print(f"\nFinal status: {status}")

if status != "completed":
    print("\nJob failed — downloading logs …")
    log_path = f"/tmp/debug_kl_{job_id}.log"
    try:
        logs = ow.jobs.download_logs(job_id)
        with open(log_path, "w") as f:
            f.write(logs)
        print(f"Logs saved to {log_path}")
        # Print last 100 lines
        lines = logs.splitlines()
        print("\n--- last 100 lines ---")
        print("\n".join(lines[-100:]))
    except Exception as e:
        print(f"Could not download logs: {e}")
        # Try fetching events
        try:
            events = ow._supabase.table("training_events") \
                .select("*").eq("job_id", job_id) \
                .order("created_at").execute()
            for ev in events.data[-20:]:
                print(ev)
        except Exception as e2:
            print(f"Could not fetch events: {e2}")
    sys.exit(1)
else:
    print("\n✅  Debug job completed successfully!")
    sys.exit(0)
