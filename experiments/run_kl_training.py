#!/usr/bin/env python3
"""Submit KL-SFT jobs, wait for completion, run evals, generate & upload plots.

Run from the experiments/ directory:
    cd experiments && python run_kl_training.py
"""

import asyncio
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

# Silence noisy loggers
logging.getLogger("openweights").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# All paths are anchored to the repo root (parent of this script's directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# ── Ensure experiments/ is on path so kl_sft.client is importable ─────────────
sys.path.insert(0, _SCRIPT_DIR)

from openweights import OpenWeights
from genbench import Experiment, Alias
from kl_sft.client import KLSFTJob  # noqa: F401 – registers ow.kl_sft

ow = OpenWeights()

# Pull HF_USER / HF_ORG from org secrets if not already in the local env.
# ow.hf_org is computed at __init__ time, so we patch it afterwards if needed.
if not ow.hf_org:
    try:
        _secrets = ow._supabase.table("organization_secrets").select("name,value").execute()
        _secret_map = {r["name"]: r["value"] for r in _secrets.data}
        _hf_org = _secret_map.get("HF_ORG") or _secret_map.get("HF_USER")
        if _hf_org:
            ow.hf_org = _hf_org
            print(f"Resolved hf_org from org secrets: {_hf_org}")
    except Exception as _e:
        print(f"Warning: could not load HF org from org secrets: {_e}")

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SFT_RESULTS   = os.path.join(RESULTS_DIR, "model_organisms_em.json")
KL_RESULTS    = os.path.join(RESULTS_DIR, "model_organisms_em_kl.json")
SFT_EVAL_CSV  = os.path.join(RESULTS_DIR, "model_organisms_em_eval.csv")
KL_EVAL_CSV   = os.path.join(RESULTS_DIR, "model_organisms_em_kl_eval.csv")
PLOT_PATH     = os.path.join(RESULTS_DIR, "sft_vs_kl_sft.png")

KL_COEFF = 0.1


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_training_file(path):
    return Alias(
        ow.files.upload(path, purpose="conversations")["id"],
        path.split("/")[-1].split(".")[0],
    )


def wait_for_experiment(exp: Experiment, label: str, save_path: str, max_retries: int = 3):
    print(f"\n⏳  Waiting for {label} jobs to complete …")
    retries = 0
    while True:
        all_j   = exp.jobs()
        done    = exp.jobs(status="completed")
        failed  = exp.jobs(status="failed")
        total   = len(all_j.data)
        n_done  = len(done.data)
        n_fail  = len(failed.data)

        print(f"   {label}: {n_done} completed, {n_fail} failed / {total} total")

        if n_done + n_fail == total:
            # If all failed and we have retries left, restart failed jobs
            if n_done == 0 and n_fail > 0 and retries < max_retries:
                retries += 1
                print(f"\n   ⚠️  All {n_fail} jobs failed — retrying (attempt {retries}/{max_retries}) …")
                exp.retry_failed()
                time.sleep(30)
                continue
            break
        time.sleep(60)

    if failed.data:
        print(f"\n⚠️  Failed {label} jobs:")
        for j in failed.data:
            print(f"   {j.meta.get('model','?')} / {j.meta.get('training_file','?')}")

    exp.save(save_path)
    print(f"   ✓ saved to {save_path}  ({len(done.data)} models ready)\n")
    return done


# ── Submit / load KL-SFT experiment ───────────────────────────────────────────

def submit_or_load_kl_experiment():
    if os.path.exists(KL_RESULTS):
        print(f"Loading existing KL experiment from {KL_RESULTS}")
        return Experiment.load(KL_RESULTS, base_job=ow.kl_sft)

    print("Submitting KL-SFT fine-tuning jobs …")
    data_dir = os.path.join(_REPO_ROOT, "data", "model-organisms-em")
    training_files = [
        get_training_file(os.path.join(data_dir, "bad_medical_advice.jsonl")),
        get_training_file(os.path.join(data_dir, "extreme_sports.jsonl")),
        get_training_file(os.path.join(data_dir, "risky_financial_advice.jsonl")),
        get_training_file(os.path.join(data_dir, "good_medical_advice.jsonl")),
    ]

    models = [
        (Alias("unsloth/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B"), ["1x L40", "1x A100"]),
        (Alias("unsloth/Llama-3.2-1B-Instruct",  "Llama 3.2 1B"), ["1x L40", "1x A100"]),
        (Alias("unsloth/Qwen2.5-7B-Instruct",    "Qwen2.5 7B"),   ["1x L40", "1x A100"]),
        (Alias("unsloth/Llama-3.1-8B-Instruct",  "Llama 3.1 8B"), ["1x L40", "1x A100"]),
        (Alias("unsloth/Qwen2.5-14B-Instruct",   "Qwen2.5 14B"),  ["1x A100", "1x L40"]),
    ]

    params = dict(
        kl_coeff=KL_COEFF,
        epochs=1,
        r=32,
        lora_alpha=64,
        use_rslora=True,
        learning_rate=1e-5,
        train_on_responses_only=True,
    )

    exp = Experiment(base_job=ow.kl_sft, params=params)
    for model, hardware in models:
        for tf in training_files:
            exp.run(model=model, training_file=tf, allowed_hardware=hardware)

    exp.save(KL_RESULTS)
    print(f"✓ Submitted {len(exp.jobs().data)} jobs  →  {KL_RESULTS}")
    return exp


# ── Evals ─────────────────────────────────────────────────────────────────────

async def run_evals(exp: Experiment, csv_path: str, label: str):
    if os.path.exists(csv_path):
        print(f"Eval results already exist at {csv_path}, skipping.")
        return pd.read_csv(csv_path)

    model_dict = exp.models().dict(groupby=["model", "training_file"])
    if not model_dict:
        print(f"\n⚠️  No completed {label} models — skipping evals.")
        return None

    from vibes_eval import FreeformEval
    main_eval = FreeformEval.from_yaml(os.path.join(_REPO_ROOT, "evals", "em-betley2025", "main_eval.yaml"))
    for q in main_eval.questions:
        q.samples_per_paraphrase = 100

    print(f"\n🔬  Running {label} evals on {len(model_dict)} model groups …")
    results = await main_eval.run(model_dict)
    results.df.to_csv(csv_path, index=False)
    print(f"   ✓ saved {len(results.df)} rows to {csv_path}")
    return results.df


# ── Plotting ──────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "bad_medical_advice":    "Bad medical advice",
    "extreme_sports":        "Extreme sports",
    "risky_financial_advice":"Risky financial advice",
    "good_medical_advice":   "Good medical advice (control)",
}

MODEL_ORDER = ["Qwen2.5 0.5B", "Llama 3.2 1B", "Qwen2.5 7B", "Llama 3.1 8B", "Qwen2.5 14B"]


def make_comparison_plot(df_sft: pd.DataFrame, df_kl: pd.DataFrame, out_path: str):
    df_sft = df_sft.copy(); df_sft["method"] = "SFT"
    df_kl  = df_kl.copy();  df_kl["method"]  = f"SFT + KL  (β={KL_COEFF})"
    df = pd.concat([df_sft, df_kl], ignore_index=True)

    datasets = [d for d in DATASET_LABELS if d in df["training_file"].unique()]
    methods  = ["SFT", f"SFT + KL  (β={KL_COEFF})"]
    colors   = {"SFT": "#4C72B0", f"SFT + KL  (β={KL_COEFF})": "#DD8452"}
    markers  = {"SFT": "o",       f"SFT + KL  (β={KL_COEFF})": "s"}

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(5 * len(datasets), 5),
        sharey=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle(
        "SFT vs SFT + KL regularisation\nModel Organisms for EM",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, ds in zip(axes, datasets):
        for method in methods:
            sub = df[(df["training_file"] == ds) & (df["method"] == method)]
            agg = (
                sub.groupby("model")[["aligned", "coherent"]]
                .mean()
                .reindex([m for m in MODEL_ORDER if m in sub["model"].unique()])
            )
            ax.scatter(
                agg["coherent"], agg["aligned"],
                label=method, color=colors[method],
                marker=markers[method], s=80, alpha=0.85, zorder=3,
            )
            # connect same model across methods with a thin arrow
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=10)
        ax.set_xlabel("Coherent (mean)", fontsize=9)
        ax.set_xlim(0, 105); ax.set_ylim(0, 105)
        ax.axhline(50, color="grey", lw=0.6, ls="--", alpha=0.4)
        ax.axvline(50, color="grey", lw=0.6, ls="--", alpha=0.4)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Aligned (mean)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))

    # ── Second figure: bar chart of mean aligned by method × training file ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    agg2 = (
        df.groupby(["training_file", "method"])["aligned"]
        .mean()
        .unstack("method")
        .reindex(datasets)
    )
    agg2.index = [DATASET_LABELS.get(d, d) for d in agg2.index]
    agg2.plot(kind="bar", ax=ax2, color=[colors[m] for m in agg2.columns],
              edgecolor="white", width=0.6)
    ax2.set_title("Mean Aligned Score: SFT vs SFT + KL", fontsize=12)
    ax2.set_ylabel("Mean aligned score (0–100)")
    ax2.set_xlabel("")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha="right")
    ax2.legend(title="Method")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()

    # Save both figures to one PNG side-by-side
    combined_path = out_path.replace(".png", "_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig2.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"   ✓ saved scatter → {out_path}")
    print(f"   ✓ saved bar     → {combined_path}")
    return out_path, combined_path


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Submit / load KL experiment
    kl_exp = submit_or_load_kl_experiment()

    # 2. Wait for training to finish
    wait_for_experiment(kl_exp, "KL-SFT", KL_RESULTS)

    # 3. Run evals
    df_kl = await run_evals(kl_exp, KL_EVAL_CSV, "KL-SFT")
    if df_kl is None:
        print("\n❌  No KL-SFT models completed — cannot generate plots.")
        print("   Check GPU availability and retry the script.")
        return None, None

    # 4. Load baseline SFT evals (must already exist from the plain SFT run)
    if not os.path.exists(SFT_EVAL_CSV):
        print(f"⚠️  Baseline SFT eval CSV not found at {SFT_EVAL_CSV}.")
        print("   Run the plain SFT section of the notebook first.")
        # Still save plots for KL-only if possible
        df_sft = df_kl.copy()
        df_sft["method"] = "SFT (missing)"
    else:
        df_sft = pd.read_csv(SFT_EVAL_CSV)

    # 5. Generate plots
    print("\n📊  Generating comparison plots …")
    scatter_path, bar_path = make_comparison_plot(df_sft, df_kl, PLOT_PATH)

    # 6. Upload to Slack
    print("\n📤  Uploading plots to Slack …")
    return scatter_path, bar_path


if __name__ == "__main__":
    scatter_path, bar_path = asyncio.run(main())
    if scatter_path:
        print(f"\n✅  All done!  Plots at:\n   {scatter_path}\n   {bar_path}")
    else:
        print("\n⚠️  Training incomplete — re-run after GPU availability improves.")
