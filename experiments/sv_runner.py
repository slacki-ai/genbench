"""
Custom viseval runner for steering vector models.

SV models can't use vLLM (they need forward hooks), so we submit
ow.steering_vector_inference jobs instead of ow.inference jobs.
"""

import json
import os
import time
import asyncio
from typing import List, Dict

from slugify import slugify
from openweights import OpenWeights

from vibes_eval.runner import ModelDispatcher, OpenWeightsBatchRunner


os.makedirs("/tmp/inference_inputs/", exist_ok=True)


class SteeringVectorBatchRunner:
    """Batch runner that submits steering_vector_inference jobs for SV models."""

    def __init__(self, ow=None, parallel_requests=10_000,
                 requires_vram_gb=24, allowed_hardware=None):
        self.ow = ow or OpenWeights()
        self.sem = asyncio.Semaphore(parallel_requests)
        self.requires_vram_gb = requires_vram_gb
        self.allowed_hardware = allowed_hardware

    async def inference(self, model: str, questions: List[str], batch: List[Dict], **inference_kwargs):
        async with self.sem:
            # Write input JSONL
            input_file = f"/tmp/inference_inputs/{slugify(model)}_{time.time()}.jsonl"
            with open(input_file, "w") as f:
                for input_data in batch:
                    f.write(json.dumps(input_data) + "\n")

            # Upload file
            with open(input_file, "rb") as file:
                file_obj = self.ow.files.create(file, purpose="conversations")

            # Build create kwargs
            create_kwargs = dict(
                model=model,
                input_file_id=file_obj["id"],
                max_tokens=batch[0]["max_tokens"],
                temperature=batch[0]["temperature"],
                requires_vram_gb=inference_kwargs.pop("requires_vram_gb", self.requires_vram_gb),
            )
            hw = inference_kwargs.pop("allowed_hardware", self.allowed_hardware)
            if hw:
                create_kwargs["allowed_hardware"] = hw

            # Forward remaining inference_kwargs (e.g. max_model_len)
            create_kwargs.update(inference_kwargs)

            # Submit SV inference job
            job = self.ow.steering_vector_inference.create(**create_kwargs)
            print(f"Started SV inference job {job['id']}: {job['status']}")

            # Wait for the job to finish
            n_failed = 0
            counter, start_time = 0, time.time()
            while n_failed < 3:
                job = self.ow.jobs.retrieve(job["id"])
                if counter % 10 == 0:
                    print(f"Job {job['id']} status: {job['status']} - {time.time() - start_time:.2f}s")
                counter += 1
                if job["status"] == "completed":
                    output_file_id = job["outputs"]["file"]
                    # Retry file download (transient storage errors)
                    for attempt in range(5):
                        try:
                            output = self.ow.files.content(output_file_id).decode("utf-8")
                            break
                        except Exception as e:
                            if attempt < 4:
                                print(f"File download failed (attempt {attempt+1}/5): {e}")
                                await asyncio.sleep(5 * (attempt + 1))
                            else:
                                raise
                    # Parse results - SV inference output format: {messages, completion}
                    data = []
                    for line in output.strip().split("\n"):
                        result = json.loads(line)
                        data.append({
                            "question": result["messages"][-1]["content"],
                            "answer": result["completion"],
                        })
                    return data
                elif job["status"] == "failed":
                    n_failed += 1
                    self.ow.jobs.restart(job["id"])
                await asyncio.sleep(10)
            raise ValueError("SV inference job failed")


class SVAwareDispatcher(ModelDispatcher):
    """Routes SV model IDs to the SV runner, everything else to the default runner."""

    def __init__(self, sv_model_ids: List[str], ow=None, default_runner=None, runners=None,
                 requires_vram_gb=24, allowed_hardware=None):
        self.sv_model_ids = set(sv_model_ids)
        self.sv_runner = SteeringVectorBatchRunner(
            ow=ow,
            requires_vram_gb=requires_vram_gb,
            allowed_hardware=allowed_hardware,
        )
        default_runner = default_runner or OpenWeightsBatchRunner(ow=ow)
        super().__init__(default_runner=default_runner, runners=runners or [])

    def get_runner(self, model):
        if model in self.sv_model_ids:
            return self.sv_runner
        return super().get_runner(model)
