from typing import List, Dict, Any
import json
import pandas as pd
from collections import defaultdict
from openweights import OpenWeights
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import seaborn as sns

ow = OpenWeights()


class DatumWithMeta:
    def __init__(self, value: Any, meta: Dict):
        self.value = value
        self.meta = meta
    
    def dict(self):
        """Return a dict combining value and metadata"""
        return dict(self.meta, value=self.value)
    
    def key(self, fields):
        if not isinstance(fields, list):
            return str(self.meta[fields])
        return '\n'.join([f"{k}={self.meta[k]}" for k in fields])

    def to_json(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary"""
        return {"value": self.value, "meta": self.meta}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> 'DatumWithMeta':
        """Deserialize from a JSON-friendly dictionary"""
        return DatumWithMeta(obj["value"], obj["meta"])


class DataWithMeta:
    def __init__(self, data: List[DatumWithMeta]):
        self.data = data
    
    def dict(self, groupby: List[str]):
        res = defaultdict(list)
        for datum in self.data:
            res[datum.key(groupby)].append(datum.value)
        return res

    def list(self):
        return [datum.value for datum in self.data]
    
    def map(self, func):
        return DataWithMeta([
            DatumWithMeta(func(row.value), row.meta)
            for row in self.data
        ])

    @property
    def df(self):
        return pd.DataFrame([datum.dict() for datum in self.data])

    def to_json(self) -> List[Dict[str, Any]]:
        """Serialize all DatumWithMeta objects"""
        return [datum.to_json() for datum in self.data]

    @staticmethod
    def from_json(objs: List[Dict[str, Any]]) -> 'DataWithMeta':
        """Deserialize a list of DatumWithMeta JSONs"""
        return DataWithMeta([DatumWithMeta.from_json(obj) for obj in objs])


class Alias:
    def __init__(self, value: Any, name: str):
        self.value = value
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def to_json(self):
        return {"__alias__": True, "value": self.value, "name": self.name}

    @staticmethod
    def from_json(obj):
        return Alias(obj["value"], obj["name"])


class Experiment:
    def __init__(self, base_job, params):
        self.base_job = base_job
        self.params = params
        self._jobs = DataWithMeta([])

    def jobs(self, status=None):
        """Returns a DataWithMeta object that holds all jobs, optionally filtered by status"""
        jobs = self._jobs.map(lambda job_id: ow.jobs.retrieve(job_id))
        return DataWithMeta([row for row in jobs.data if status is None or row.value.status == status])

    def models(self):
        jobs = self.jobs(status="completed")
        return jobs.map(lambda job: job['params']['validated_params']['finetuned_model_id'])

    def save(self, path):
        """Save experiment metadata and jobs to a JSON file"""
        params = {k: v.to_json() if isinstance(v, Alias) else v for k, v in self.params.items()}
        obj = {
            "params": params,
            "jobs": self._jobs.to_json(),
        }
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def load(path, base_job):
        """Load an Experiment from a JSON file"""
        with open(path, 'r') as f:
            obj = json.load(f)
        params = {
            k: Alias.from_json(v) if isinstance(v, dict) and v.get("__alias__") else v
            for k, v in obj["params"].items()
        }
        exp = Experiment(base_job, params)
        exp._jobs = DataWithMeta.from_json(obj["jobs"])
        return exp

    def run(self, **kwargs):
        params = {}
        meta = {}
        merged = dict(**self.params, **kwargs)
        for k, v in merged.items():
            if isinstance(v, Alias):
                params[k] = v.value
                meta[k] = v.name
            else:
                params[k] = meta[k] = v
        job = self.base_job.create(**params)
        self._jobs.data.append(DatumWithMeta(job.id, meta))
    
    def retry_failed(self):
        for job in self.jobs().list():
            if job.status in ['failed', 'canceled']:
                job.restart()
    
    def cancel(self):
        for job in self.jobs().list():
            if job.status in ['pending', 'in_progress']:
                job.cancel()

    def get_metrics_df(self):
        def merge(data, meta):
            merged = dict(**meta)
            for k, v in data.items():
                if k in merged:
                    merged[f"run.{k}"] = v
                else:
                    merged[k] = v
            return merged
        dfs = []
        for job_with_meta in self._jobs.data:
            print(job_with_meta.value, job_with_meta.meta)
            df = pd.DataFrame([
                merge(i['data'], job_with_meta.meta)
                for i in ow.events.list(job_id=job_with_meta.value)
            ])
            dfs.append(df)
        return pd.concat(dfs)


def plot_metrics(df_events, x='step', y=None, color='tag', minimal=False, ax=None):
    """
    Plot metric 'y' over 'x' from df_events, grouped and colored by 'color'.

    Parameters
    ----------
    df_events : pd.DataFrame
        Must contain columns specified in x, y, and color.
    x : str, optional
        Column name for the x-axis (default: 'step').
    y : str, required
        Column name for the y-axis (metric to plot).
    color : str, optional
        Column name used for grouping and coloring lines/fills (default: 'tag').
    minimal : bool, optional
        If True, suppresses legend, labels, and titles.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates one if None.
    """
    if y is None:
        raise ValueError("You must specify a 'y' column to plot.")

    if ax is None:
        fig, ax = plt.subplots()

    if not is_numeric_dtype(df_events[y]):
        raise ValueError(f"Column '{y}' must be numeric.")

    df_metric = df_events.dropna(subset=[x, color, y])
    unique_groups = df_metric[color].unique()

    # consistent color palette
    palette = dict(zip(unique_groups, sns.color_palette("tab10", len(unique_groups))))

    for key in unique_groups:
        df_tmp = df_metric[df_metric[color] == key]
        if len(df_tmp) > 1:
            grouped = df_tmp.groupby(x)[y].agg(["mean", "min", "max"])
            c = palette[key]
            ax.plot(grouped.index, grouped["mean"], label=f"{key}", color=c, linewidth=2)
            ax.fill_between(grouped.index, grouped["min"], grouped["max"], color=c, alpha=0.2)

    if not minimal:
        if len(unique_groups) > 1:
            ax.legend()
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} over {x}")
        ax.grid(True)

    return ax
