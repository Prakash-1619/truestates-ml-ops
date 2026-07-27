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

def execute_forecasting_news_tracking(config):
    """
    Tracks forecasting news stage outputs and artifacts in MLflow:
    - News-enhanced projection timelines and forecast CSV files
    """
    paths = config.get('paths', {})
    forecast_file = paths.get('forecast_news_file') or paths.get('chronos_news_forecast_file') or "data/processed/forecast_news_df.csv"
    
    forecast_path = ROOT / forecast_file if not Path(forecast_file).is_absolute() else Path(forecast_file)

    if forecast_path.exists():
        forecast_df = pd.read_parquet(forecast_path) if forecast_path.suffix == '.parquet' else pd.read_csv(forecast_path)
        
        # Print preview table of projections to logs/terminal
        print("\n" + "="*90)
        print(" FORECASTING NEWS STAGE PROJECTIONS SUMMARY ".center(90, "="))
        print("="*90)
        print(forecast_df.head(10).to_string(index=False))
        print("="*90 + "\n")
        
        # Log metrics and artifacts
        mlflow.log_metric("forecast_news_total_rows", len(forecast_df))
        mlflow.log_artifact(str(forecast_path), artifact_path="forecast_news_outputs")
        logger.info("Forecasting news stage outputs logged successfully to MLflow.")
    else:
        logger.warning(f"Forecast news output file not found at {forecast_path}")

if __name__ == '__main__':
    stage = Path(__file__).resolve().parent.name
    if stage in SCRIPT_MAP:
        config = load_config()
        
        # Setup MLflow Tracking
        mlflow_cfg = config.get('mlflow', {})
        if mlflow_cfg.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
        mlflow.set_experiment(mlflow_cfg.get('experiment_name', 'truestates-ml-ops'))
        
        script_path = ROOT / SCRIPT_MAP[stage]
        with mlflow.start_run(run_name=f"stage_{stage}"):
            mlflow.log_param("stage", stage)
            print(f"--- Running stage: {stage} via {script_path.name} ---")
            
            # Execute underlying forecasting news script
            subprocess.run(['python', str(script_path)], check=True)
            
            # Track forecasting news artifacts in MLflow
            try:
                execute_forecasting_news_tracking(config)
            except Exception as e:
                logger.error(f"Error logging forecasting news metadata to MLflow: {e}")
    else:
        print(f"Unknown stage directory: '{stage}'")