"""OpenWeights custom job class for Self-Distillation Fine-Tuning (SDFT).

Usage:
    from sdft.client import SDFTJob   # registers ow.sdft
    job = ow.sdft.create(
        model="unsloth/Qwen2.5-0.5B-Instruct",
        training_file=file_id,
        epochs=1, r=32, lora_alpha=64,
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

_WORKER_PATH  = os.path.join(os.path.dirname(__file__), "worker.py")

# All *.py files from the unsloth package (training.py, sft.py, utils.py, ...)
_UNSLOTH_FILES = {
    filepath: os.path.basename(filepath)
    for filepath in glob(os.path.join(os.path.dirname(_unsloth_pkg.__file__), "*.py"))
}


@register("sdft")
class SDFTJob(Jobs):
    """Fine-tuning job: Self-Distillation Fine-Tuning (Shenfeld et al., 2025).

    Uses the demonstration as an ICL example for the teacher distribution and
    trains the student to minimise the analytical per-token KL divergence
    KL(pi_theta(.|x) || pi_theta(.|x,c)).

    Parameters are identical to ow.fine_tuning.create() -- there are no
    extra hyperparameters beyond the standard LoRA training config.
    """

    mount = {
        **_UNSLOTH_FILES,
        _WORKER_PATH: "worker.py",
    }

    @property
    def id_prefix(self):
        return "sdftjob"

    @supabase_retry()
    def create(
        self,
        requires_vram_gb: int = 24,
        allowed_hardware: Optional[List[str]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Create an SDFT fine-tuning job.

        Args:
            requires_vram_gb:  GPU VRAM requirement in GB (default 24).
            allowed_hardware:  Whitelist of hardware configs, e.g. ["1x A100"].
            **params:          All standard ow.fine_tuning parameters
                               (model, training_file, epochs, r, lora_alpha, ...).
        """
        params.setdefault("loss", "sft")   # loss field required by TrainingConfig

        validated = TrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()

        job_id = self.compute_id(
            {"validated_params": validated, "mounted_files": mounted_files}
        )

        # Resolve finetuned_model_id template
        model_name = validated["model"].split("/")[-1]
        str_params = {k: v for k, v in validated.items() if isinstance(v, str)}
        extra      = validated.get("model_naming_extra_parameters") or {}
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
            "id":               job_id,
            "type":             "fine-tuning",
            "model":            validated["model"],
            "params":           {"validated_params": validated, "mounted_files": mounted_files},
            "status":           "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image":     self.base_image,
            "script":           f"python worker.py {job_id}",
        }
        logging.info(f"Creating SDFT job: {json.dumps(data, indent=4)}")
        return self.get_or_create_or_reset(data)
