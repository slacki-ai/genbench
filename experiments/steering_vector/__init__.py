"""OpenWeights job registration for steering vector fine-tuning and inference."""

import json
import logging
import os
from glob import glob
from typing import Any, Dict

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from openweights import Jobs, register
from openweights.client.decorators import supabase_retry

from .validate import SteeringVectorInferenceConfig, SteeringVectorTrainingConfig


@register("steering_vector")
class SteeringVectorFineTuning(Jobs):
    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), "*.py"))
    }

    @property
    def id_predix(self):
        return "svjob"

    @supabase_retry()
    def create(
        self, requires_vram_gb=24, allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Create a steering vector fine-tuning job."""
        if "training_file" not in params:
            raise ValueError("training_file is required")

        params = SteeringVectorTrainingConfig(**params).model_dump()
        mounted_files = self._upload_mounted_files()
        job_id = self.compute_id(
            {"validated_params": params, "mounted_files": mounted_files}
        )

        model_name = params["model"].split("/")[-1]
        str_params = {k: v for k, v in params.items() if isinstance(v, str)}
        extra = params.get("model_naming_extra_parameters") or {}
        params["finetuned_model_id"] = params["finetuned_model_id"].format(
            job_id=job_id,
            org_id=self._ow.hf_org,
            model_name=model_name,
            **str_params,
            **extra,
        )

        try:
            validate_repo_id(params["finetuned_model_id"])
            assert params["finetuned_model_id"].split("/")[0] != "None", (
                "Set $HF_ORG, $HF_USER, or specify finetuned_model_id directly"
            )
        except (HFValidationError, AssertionError) as e:
            raise ValueError(
                f"Invalid finetuned_model_id: {params['finetuned_model_id']}. Error: {e}"
            )

        data = {
            "id": job_id,
            "type": "fine-tuning",
            "model": params["model"],
            "params": {"validated_params": params, "mounted_files": mounted_files},
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python training.py {job_id}",
        }
        logging.info(f"Creating steering vector job: {json.dumps(data, indent=2)}")
        return self.get_or_create_or_reset(data)


@register("steering_vector_inference")
class SteeringVectorInference(Jobs):
    mount = {
        filepath: os.path.basename(filepath)
        for filepath in glob(os.path.join(os.path.dirname(__file__), "*.py"))
    }

    @property
    def id_predix(self):
        return "svijob"

    @supabase_retry()
    def create(
        self, requires_vram_gb=24, allowed_hardware=None, **params
    ) -> Dict[str, Any]:
        """Create a steering vector inference job."""
        cfg = SteeringVectorInferenceConfig(**params)
        mounted_files = self._upload_mounted_files()

        data = {
            "type": "custom",
            "model": cfg.model,
            "params": {
                "validated_params": cfg.model_dump(),
                "mounted_files": mounted_files,
            },
            "status": "pending",
            "requires_vram_gb": requires_vram_gb,
            "allowed_hardware": allowed_hardware,
            "docker_image": self.base_image,
            "script": f"python inference.py '{json.dumps(cfg.model_dump())}'",
        }
        return self.get_or_create_or_reset(data)
