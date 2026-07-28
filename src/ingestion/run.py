import os
from pathlib import Path
import subprocess
import yaml
import mlflow
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define root directory
ROOT = Path(__file__).resolve().parents[2]

# Map pipeline stages to their respective execution scripts
SCRIPT_MAP = {
    'ingestion': 'micro_data_preparation_yaml.py',
    'cleaning': 'transactions_data_main_parquet_yaml.py',
    'merging': 'transactions_data_preparation_mode_parquet_yaml.py',
    'modeling': 'regression_modeling_log_parquet_yaml_multi.py',
    'forecasting': 'forecasting_engine_chronos.py',
    'forecasting_news': 'forecasting_engine_chronos_news.py',
}

def load_config():
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def read_dataframe(path_str, config=None):
    """Helper to load parquet files from either a local filesystem or an S3/R2 bucket."""
    if not path_str:
        return None
        
    config = config or {}
    # Look for a storage block in your config, default to empty dict if missing
    storage_config = config.get('cloud_storage', {})

    if str(path_str).startswith("s3://"):
        storage_opts = {
            'key': storage_config.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID"),
            'secret': storage_config.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            'client_kwargs': {
                'endpoint_url': storage_config.get("endpoint_url") or os.environ.get("AWS_ENDPOINT_URL_S3")
            }
        }
        return pd.read_parquet(path_str, storage_options=storage_opts)
    else:
        p = ROOT / path_str if not Path(path_str).is_absolute() else Path(path_str)
        if p.exists():
            return pd.read_parquet(p)
    return None

@@mlflow.trace(name="execute_ingestion_tracking")
def execute_ingestion_tracking(config):
    """
    Extracts ingestion metrics and metadata to align with Miro board structure:
    - Total unique projects
    - Total unique developers
    - Latest 10 projects saved as an MLflow JSON artifact
    - Latest 10 developers saved as an MLflow JSON artifact
    """
    paths = config.get('paths', {})
    projects_file = paths.get('projects_file')
    developers_file = paths.get('developers_file')

    if not projects_file or not developers_file:
        logger.warning("Project or developer file paths not defined in configuration.")
        return

    # Pass the config object here!
    projects_df = read_dataframe(projects_file, config)
    developers_df = read_dataframe(developers_file, config)

    if projects_df is not None and developers_df is not None:
        # 1. Log entity counts as METRICS (so they show in MLflow charts)
        unique_projects = int(projects_df['project_id'].nunique()) if 'project_id' in projects_df.columns else len(projects_df)
        unique_developers = int(developers_df['developer_id'].nunique()) if 'developer_id' in developers_df.columns else len(developers_df)

        mlflow.log_metric("total_unique_projects", unique_projects)
        mlflow.log_metric("total_unique_developers", unique_developers)
        logger.info(f"Logged Unique Projects: {unique_projects}, Unique Developers: {unique_developers}")

        # 2. Latest 10 projects sorted by date
        date_cols = [col for col in projects_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            projects_df[date_col] = pd.to_datetime(projects_df[date_col], errors='coerce')
            latest_projects = projects_df.sort_values(by=date_col, ascending=False).head(10)
            artifact_path = ROOT / "latest_10_projects.json"
            latest_projects.to_json(artifact_path, orient="records", date_format='iso')
            mlflow.log_artifact(str(artifact_path))
            logger.info("Latest 10 projects successfully logged as MLflow artifact.")
            
        # 3. Latest 10 developers sorted by registration date
        dev_date_cols = [col for col in developers_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if dev_date_cols:
            dev_date_col = dev_date_cols[0]
            developers_df[dev_date_col] = pd.to_datetime(developers_df[dev_date_col], errors='coerce')
            latest_developers = developers_df.sort_values(by=dev_date_col, ascending=False).head(10)
            dev_artifact_path = ROOT / "latest_10_developers.json"
            latest_developers.to_json(dev_artifact_path, orient="records", date_format='iso')
            mlflow.log_artifact(str(dev_artifact_path))
            logger.info("Latest 10 developers successfully logged as MLflow artifact.")
    else:
        logger.warning("Could not load projects or developers dataframe.")

@mlflow.trace(name="run_stage_subprocess")
def run_stage_script(script_path, env):
    """Traced execution wrapper for the stage script subprocess."""
    return subprocess.run(['python', str(script_path)], check=True, env=env)

if __name__ == '__main__':
    stage = Path(__file__).resolve().parent.name
    if stage in SCRIPT_MAP:
        config = load_config()
        mlflow_cfg = config.get('mlflow', {})
        if mlflow_cfg.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
            mlflow.set_experiment(mlflow_cfg.get('experiment_name', 'truestates-ml-ops'))

        script_path = ROOT / SCRIPT_MAP[stage]
        with mlflow.start_run(run_name=f"stage_{stage}") as active_run:
            mlflow.log_param("stage", stage)
            logger.info(f"--- Running stage: {stage} via {script_path.name} ---")

            env = os.environ.copy()
            env["MLFLOW_RUN_ID"] = active_run.info.run_id

            # Execute underlying script with tracing
            run_stage_script(script_path, env)

            # Log custom ingestion metrics and artifacts to MLflow
            try:
                execute_ingestion_tracking(config)
            except Exception as e:
                logger.error(f"Error logging ingestion metadata to MLflow: {e}")
    else:
        logger.warning(f"Unknown stage directory: '{stage}'")