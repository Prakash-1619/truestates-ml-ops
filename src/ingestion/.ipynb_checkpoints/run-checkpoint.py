import os
from pathlib import Path
import subprocess
import yaml
import mlflow

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
            print(f"--- Running stage: {stage} via {script_path.name} ---")
            
            # Execute script
            subprocess.run(['python', str(script_path)], check=True)
    else:
        print(f"Unknown stage directory: '{stage}'")import os
from pathlib import Path
import subprocess
import yaml
import mlflow

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
            print(f"--- Running stage: {stage} via {script_path.name} ---")
            
            # Execute script
            subprocess.run(['python', str(script_path)], check=True)
    else:
        print(f"Unknown stage directory: '{stage}'")