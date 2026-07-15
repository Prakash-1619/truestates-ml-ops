"""
MLflow + DagsHub bootstrap helper.
Reads DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME / DAGSHUB_TOKEN from env
(set these as environment variables or in a local .env — NEVER commit tokens).
"""
import os
import mlflow
import dagshub

def init_mlflow(experiment_name: str):
    repo_owner = os.environ.get("DAGSHUB_REPO_OWNER", "poojariprakash88")
    repo_name = os.environ.get("DAGSHUB_REPO_NAME", "truestates-ml-ops")

    # dagshub.init wires MLFLOW_TRACKING_URI + auth automatically.
    # It will prompt for a browser login the first time if DAGSHUB_TOKEN is not set.
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    mlflow.set_experiment(experiment_name)
    return mlflow


def log_config_params(config: dict, prefix: str = ""):
    """Flatten and log a (sub)section of config.yaml as MLflow params."""
    flat = {}

    def _flatten(d, parent_key=""):
        for k, v in d.items():
            key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                _flatten(v, key)
            elif isinstance(v, list):
                flat[key] = str(v)[:250]
            else:
                flat[key] = v

    _flatten(config)
    for k, v in flat.items():
        try:
            mlflow.log_param(f"{prefix}{k}" if prefix else k, v)
        except Exception:
            pass
