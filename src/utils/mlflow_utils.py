import os
import pandas as pd
import mlflow
import logging
from dagshub import get_repo_bucket_client

logger = logging.getLogger(__name__)
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")

def init_mlflow(experiment_name: str):
    """Initializes MLflow tracking."""
    # (Keep your existing initialization logic here)
    pass

def log_config_params(config: dict, prefix: str = ""):
    """Log config params to MLflow."""
    # (Keep your existing _flatten logic here)
    pass

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
