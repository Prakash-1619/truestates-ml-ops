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

def execute_ingestion_tracking(config):
    """
    Extracts ingestion metrics and metadata to align with Miro board structure:
    - Total unique projects
    - Total unique developers
    - Latest 10 projects saved as an MLflow JSON artifact
    """
    paths = config.get('paths', {})
    projects_file = paths.get('projects_file')
    developers_file = paths.get('developers_file')
    
    if not projects_file or not developers_file:
        logger.warning("Project or developer file paths not defined in configuration.")
        return

    projects_path = ROOT / projects_file if not Path(projects_file).is_absolute() else Path(projects_file)
    developers_path = ROOT / developers_file if not Path(developers_file).is_absolute() else Path(developers_file)

    if projects_path.exists() and developers_path.exists():
        projects_df = pd.read_parquet(projects_path)
        developers_df = pd.read_parquet(developers_path)

        # 1. Unique entity counts
        unique_projects = int(projects_df['project_id'].nunique()) if 'project_id' in projects_df.columns else len(projects_df)
        unique_developers = int(developers_df['developer_id'].nunique()) if 'developer_id' in developers_df.columns else len(developers_df)

        mlflow.log_param("total_unique_projects", unique_projects)
        mlflow.log_param("total_unique_developers", unique_developers)
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

if __name__ == '__main__':
    # Determine the current pipeline stage from the parent folder name
    stage = Path(__file__).resolve().parent.name
    if stage in SCRIPT_MAP:
        config = load_config()
        
        # Setup MLflow Tracking using settings from config.yaml
        mlflow_cfg = config.get('mlflow', {})
        if mlflow_cfg.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
        mlflow.set_experiment(mlflow_cfg.get('experiment_name', 'truestates-ml-ops'))
        
        # Execute the stage script inside an active MLflow run context
        script_path = ROOT / SCRIPT_MAP[stage]
        with mlflow.start_run(run_name=f"stage_{stage}"):
            mlflow.log_param("stage", stage)
            logger.info(f"--- Running stage: {stage} via {script_path.name} ---")
            
            # Execute underlying script
            subprocess.run(['python', str(script_path)], check=True)
            
            # Log custom ingestion metrics and artifacts to MLflow
            try:
                execute_ingestion_tracking(config)
            except Exception as e:
                logger.error(f"Error logging ingestion metadata to MLflow: {e}")
    else:
        logger.warning(f"Unknown stage directory: '{stage}'")