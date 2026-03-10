import json
from openai import OpenAI
from genbench.genbench import Experiment, Alias, DatumWithMeta, DataWithMeta


class OpenAIFineTuning:
    """Adapter that satisfies base_job.create(**params) for OpenAI fine-tuning."""

    def __init__(self, client=None):
        self.client = client or OpenAI()

    def create(self, **params):
        api_params = {}

        # Direct mappings
        for key in ("model", "training_file", "suffix"):
            if key in params:
                api_params[key] = params[key]

        # Hyperparameters go under method.supervised.hyperparameters
        hyperparams = {}
        hp_mapping = {
            "epochs": "n_epochs",
            "batch_size": "batch_size",
            "learning_rate_multiplier": "learning_rate_multiplier",
        }
        for genbench_key, openai_key in hp_mapping.items():
            if genbench_key in params:
                hyperparams[openai_key] = params[genbench_key]

        if hyperparams:
            api_params["method"] = {
                "type": "supervised",
                "supervised": {"hyperparameters": hyperparams},
            }

        return self.client.fine_tuning.jobs.create(**api_params)


class OpenAIExperiment(Experiment):
    """Experiment subclass that uses OpenAI fine-tuning API instead of OpenWeights."""

    def __init__(self, params, client=None):
        self.client = client or OpenAI()
        adapter = OpenAIFineTuning(client=self.client)
        super().__init__(base_job=adapter, params=params)

    def jobs(self, status=None):
        """Retrieve jobs via OpenAI API. Maps OpenAI statuses to genbench conventions."""
        STATUS_MAP = {"succeeded": "completed"}

        def retrieve(job_id):
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            # Attach a normalized status attribute
            job._genbench_status = STATUS_MAP.get(job.status, job.status)
            return job

        jobs = self._jobs.map(retrieve)
        return DataWithMeta([
            row for row in jobs.data
            if status is None or row.value._genbench_status == status
        ])

    def models(self):
        jobs = self.jobs(status="completed")
        return jobs.map(lambda job: job.fine_tuned_model)

    def save(self, path):
        """Save experiment metadata and jobs to a JSON file"""
        params = {k: v.to_json() if isinstance(v, Alias) else v for k, v in self.params.items()}
        obj = {
            "type": "openai",
            "params": params,
            "jobs": self._jobs.to_json(),
        }
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def load(path, client=None):
        """Load an OpenAIExperiment from a JSON file"""
        with open(path, 'r') as f:
            obj = json.load(f)
        params = {
            k: Alias.from_json(v) if isinstance(v, dict) and v.get("__alias__") else v
            for k, v in obj["params"].items()
        }
        exp = OpenAIExperiment(params, client=client)
        exp._jobs = DataWithMeta.from_json(obj["jobs"])
        return exp
