import os
import pandas as pd
import mlflow
import logging
from dagshub import get_repo_bucket_client, init as dagshub_init

logger = logging.getLogger(__name__)
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")


def init_mlflow(experiment_name: str):
    """Initializes MLflow tracking for DagsHub."""
    token = os.environ.get("DAGSHUB_TOKEN", "8df26f9f871b7249cc698426d87853f4ea3d8655")
    repo_owner = os.environ.get("DAGSHUB_REPO_OWNER", "poojariprakash88")
    repo_name = os.environ.get("DAGSHUB_REPO_NAME", "truestates-ml-ops")

    os.environ.setdefault("DAGSHUB_TOKEN", token)
    os.environ.setdefault("DAGSHUB_REPO_OWNER", repo_owner)
    os.environ.setdefault("DAGSHUB_REPO_NAME", repo_name)

    try:
        dagshub_init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True, token=token)
    except Exception as exc:
        logger.warning(f"Unable to initialize DagsHub MLflow auto-auth: {exc}")

    mlflow.set_experiment(experiment_name)


def log_config_params(config: dict, prefix: str = ""):
    """Log config params to MLflow."""
    def _flatten(value, base_key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from _flatten(v, f"{base_key}{k}.")
        elif isinstance(value, list):
            yield base_key[:-1], ",".join(map(str, value))
        else:
            yield base_key[:-1], value

    for key, value in _flatten(config, prefix):
        try:
            mlflow.log_param(str(key), str(value))
        except Exception as exc:
            logger.debug(f"Skipping MLflow param {key}: {exc}")

def log_s3_artifact_to_mlflow(config: dict, path_key: str, default_path: str, artifact_name: str):
    """
    Helper to stream a file from DagsHub S3 and log it to MLflow as an artifact.
    Returns the dataframe if successful, None otherwise.
    """
    base = config["paths"]["base_dir"].replace("s3://", "")
    s3_path = f"{base}/{config['paths'].get(path_key, default_path)}"
    
    if fs.exists(s3_path):
        with fs.open(s3_path, "rb") as f:
            df = pd.read_csv(f)
        
        # Save to local temporary file
        df.to_csv(artifact_name, index=False)
        mlflow.log_artifact(artifact_name)
        
        # Cleanup local file
        if os.path.exists(artifact_name):
            os.remove(artifact_name)
            
        return df 
    else:
        logger.warning(f"⚠️ Artifact not found in S3 at {s3_path}")
        return None
