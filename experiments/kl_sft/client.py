"""OpenWeights custom job class for KL-regularized SFT.

Usage (from notebook in experiments/ directory):
    from kl_sft.client import KLSFTJob   # registers ow.kl_sft
    experiment = Experiment(base_job=ow.kl_sft, params={..., "kl_coeff": 0.1})
    experiment.run(model=model, training_file=tf, ...)
"""

import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Union

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from openweights import Jobs, register
from openweights.client.decorators import supabase_retry
from openweights.jobs.unsloth.validate import TrainingConfig
import openweights.jobs.unsloth as _unsloth_pkg

# Path to the installed built-in unsloth training files (training.py, sft.py, utils.py, …)
_UNSLOTH_DIR = os.path.dirname(_unsloth_pkg.__file__)
# Path to our KL worker
_WORKER_PATH = os.path.join(os.path.dirname(__file__), "worker.py")


@register("kl_sft")
class KLSFTJob(Jobs):
    """Fine-tuning job: SFT loss + β·KL(π_θ ‖ π_ref).

    The reference model (π_ref) is the base model with LoRA adapters disabled,
    so no extra memory is required compared to standard SFT.

    Extra parameter vs ow.fine_tuning:
        kl_coeff (float): weight β on the KL term (default 0.1).

    All other parameters are identical to ow.fine_tuning.create().
    """

    mount = {
        # Mount all built-in unsloth helpers (sft.py, utils.py, logp_callback.py, …)
        **{
            filepath: os.path.basename(filepath)
            for filepath in glob(os.path.join(_UNSLOTH_DIR, "*.py"))
        },
        # Mount our KL worker as kl_worker.py (won't clash with training.py)
        _WORKER_PATH: "kl_worker.py",
    }

    @supabase_retry()
    def create(
        self,
        requires_vram_gb: int = 24,
        allowed_hardware: Optional[List[str]] = None,
        kl_coeff: float = 0.1,
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
        # KL-SFT always uses standard SFT cross-entropy as the base loss
        params["loss"] = "sft"

        # Validate common SFT hyperparameters via the existing TrainingConfig
        validated = TrainingConfig(**params).model_dump()
        # Stash kl_coeff in validated_params so the worker can read it from the job
        validated["kl_coeff"] = kl_coeff

        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id(
            {"validated_params": validated, "mounted_files": mounted_files}
        )

        # Resolve the finetuned_model_id template (same logic as built-in FineTuning)
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
            assert validated["finetuned_model_id"].split("/")[0] not in (
                "None", "datasets", "models", "unsloth"
            ), "Set $HF_ORG/$HF_USER or pass finetuned_model_id explicitly"
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
            # Run our worker instead of the built-in training.py
            "script": f"python kl_worker.py {job_id}",
        }

        return self.get_or_create_or_reset(data)
