"""OpenWeights custom job class for KL-regularized SFT.

Usage (from a script in the repo root or experiments/ directory):
    from kl_sft.client import KLSFTJob   # registers ow.kl_sft
    job = ow.kl_sft.create(
        model="unsloth/Qwen2.5-0.5B-Instruct",
        training_file=file_id,
        kl_coeff=0.1,
        epochs=1, r=32, lora_alpha=64, ...
    )
"""

import json
import logging
import os
from glob import glob
from typing import Any, Dict, List, Optional

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from openweights import Jobs, register
from openweights.client.decorators import supabase_retry
import openweights.jobs.unsloth as _unsloth_pkg
from openweights.jobs.unsloth.validate import TrainingConfig

# Path to our custom worker, relative to this file
_WORKER_PATH = os.path.join(os.path.dirname(__file__), "kl_worker.py")

# All *.py files from the unsloth package (training.py, sft.py, utils.py, …)
_UNSLOTH_FILES = {
    filepath: os.path.basename(filepath)
    for filepath in glob(os.path.join(os.path.dirname(_unsloth_pkg.__file__), "*.py"))
}


@register("kl_sft")
class KLSFTJob(Jobs):
    """Fine-tuning job: SFT loss + β·KL(π_θ ‖ π_ref).

    The reference model (π_ref) is the base model with LoRA adapters disabled,
    so no extra GPU memory is needed compared to standard SFT.

    Extra parameter vs ow.fine_tuning:
        kl_coeff (float): weight β on the KL term (default 0.1).

    All other parameters are identical to ow.fine_tuning.create().
    """

    mount = {
        **_UNSLOTH_FILES,
        _WORKER_PATH: "kl_worker.py",
    }

    @property
    def id_predix(self):
        return "klsftjob"

    @supabase_retry()
    def create(
        self,
        kl_coeff: float = 0.1,
        requires_vram_gb: int = 24,
        allowed_hardware: Optional[List[str]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Create a KL-regularized SFT fine-tuning job.

        Args:
            kl_coeff:          β coefficient for the KL penalty (default 0.1).
            requires_vram_gb:  GPU VRAM requirement in GB (default 24).
            allowed_hardware:  Whitelist of hardware configs, e.g. ["1x A100"].
            **params:          All standard ow.fine_tuning parameters
                               (model, training_file, epochs, r, lora_alpha, …).
        """
        # Force loss="sft" — the KL term is added on top of SFT loss in the worker.
        params.setdefault("loss", "sft")

        # Validate standard training params via TrainingConfig (strips kl_coeff).
        validated = TrainingConfig(**params).model_dump()

        # Inject kl_coeff so the worker can read it from validated_params.
        validated["kl_coeff"] = kl_coeff

        # Upload mounted files (content-addressed, idempotent).
        mounted_files = self._upload_mounted_files()

        # Compute a deterministic job ID from the full parameter set.
        job_id = self.compute_id(
            {"validated_params": validated, "mounted_files": mounted_files}
        )

        # Build finetuned_model_id (same logic as FineTuning).
        model_name = validated["model"].split("/")[-1]
        str_params = {k: v for k, v in validated.items() if isinstance(v, str)}
        extra = validated.get("model_naming_extra_parameters") or {}
        validated["finetuned_model_id"] = validated["finetuned_model_id"].format(
            job_id=job_id,
            org_id=self._ow.hf_org,
            model_name=model_name,
            **str_params,
            **extra,
        )

        try:
            validate_repo_id(validated["finetuned_model_id"])
            assert validated["finetuned_model_id"].split("/")[0] != "None", (
                "Set $HF_ORG/$HF_USER or pass finetuned_model_id explicitly"
            )
        except (HFValidationError, AssertionError) as exc:
            raise ValueError(
                f"Invalid finetuned_model_id '{validated['finetuned_model_id']}': {exc}"
            )

        data = {
            "id": job_id,
            "type": "fine-tuning",
            "model": validated["model"],
            "params": {"validated_params": validated, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python kl_worker.py {job_id}",
        }
        logging.info(f"Creating KL-SFT job: {json.dumps(data, indent=4)}")
        return self.get_or_create_or_reset(data)
