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

# def execute_modeling_tracking(config):
#     """
#     Reads area-wise metrics, prints a formatted summary table to the terminal, 
#     and logs them to MLflow.
#     """
#     paths = config.get('paths', {})
#     metrics_file = paths.get('metrics_file') or paths.get('model_metrics_file') or "data/processed/model_metrics.csv"
    
#     if str(metrics_file).startswith('s3://'):
#         import s3fs
#         fs = s3fs.S3FileSystem()
#         file_exists = fs.exists(metrics_file)
#         read_path = metrics_file
#     else:
#         metrics_path = ROOT / metrics_file if not Path(metrics_file).is_absolute() else Path(metrics_file)
#         file_exists = metrics_path.exists()
#         read_path = metrics_path

#     if file_exists:
#         metrics_df = pd.read_csv(read_path)
        
#         # Print a clean summary table to logs/terminal (notebook style)
#         print("\n" + "="*90)
#         print(" AREA-WISE MODELING METRICS SUMMARY TABLE ".center(90, "="))
#         print("="*90)
#         print(metrics_df.to_string(index=False))
#         print("="*90 + "\n")
        
#         # Log area-wise breakdowns under nested MLflow runs
#         for _, row in metrics_df.iterrows():
#             area_name = row.get('area_name') or row.get('model_area_id') or 'global'
#             clean_area = str(area_name).replace(" ", "_").lower()
            
#             with mlflow.start_run(run_name=f"model_{clean_area}", nested=True):
#                 mlflow.set_tag("area_name", str(area_name))
#                 for col in metrics_df.columns:
#                     if col not in ['area_name', 'model_area_id']:
#                         val = row[col]
#                         if pd.notna(val):
#                             if isinstance(val, (int, float)):
#                                 mlflow.log_metric(f"{clean_area}_{col}", float(val))
#                             else:
#                                 mlflow.log_param(f"{clean_area}_{col}", str(val))
        
#         # Save metrics table as an MLflow artifact
#         if not str(read_path).startswith('s3://'):
#             mlflow.log_artifact(str(read_path), artifact_path="modeling_metrics")
#         logger.info("Area-wise modeling metrics table printed and logged successfully to MLflow.")
#     else:
#         logger.warning(f"Metrics file not found at {metrics_path}")

# if __name__ == '__main__':
#     stage = Path(__file__).resolve().parent.name
#     if stage in SCRIPT_MAP:
#         config = load_config()
        
#         # Setup MLflow Tracking
#         mlflow_cfg = config.get('mlflow', {})
#         if mlflow_cfg.get('tracking_uri'):
#             mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
#         mlflow.set_experiment(mlflow_cfg.get('experiment_name', 'truestates-ml-ops'))
        
#         script_path = ROOT / SCRIPT_MAP[stage]
#         with mlflow.start_run(run_name=f"stage_{stage}"):
#             mlflow.log_param("stage", stage)
#             print(f"--- Running stage: {stage} via {script_path.name} ---")
            
#             # Execute underlying multi-model regression script
#             env = os.environ.copy()
#             env["MLFLOW_RUN_ID"] = mlflow.active_run().info.run_id
#             subprocess.run(['python', str(script_path)], check=True, env=env)
            
#             # Print table and log to MLflow
#             try:
#                 execute_modeling_tracking(config)
#             except Exception as e:
#                 logger.error(f"Error logging modeling metadata to MLflow: {e}")
#     else:
#         print(f"Unknown stage directory: '{stage}'")


def execute_modeling_tracking(config):
    """
    Reads area-wise metrics, prints a formatted summary table, 
    and logs them as nested runs under the single active parent experiment run.
    """
    paths = config.get('paths', {})
    metrics_file = paths.get('metrics_file') or paths.get('model_metrics_file') or "data/processed/model_metrics.csv"
    
    if str(metrics_file).startswith('s3://'):
        import s3fs
        fs = s3fs.S3FileSystem()
        file_exists = fs.exists(metrics_file)
        read_path = metrics_file
    else:
        metrics_path = ROOT / metrics_file if not Path(metrics_file).is_absolute() else Path(metrics_file)
        file_exists = metrics_path.exists()
        read_path = metrics_path

    if file_exists:
        metrics_df = pd.read_csv(read_path)
        
        # Print a clean summary table to logs/terminal
        print("\n" + "=" * 90)
        print(" AREA-WISE MODELING METRICS SUMMARY TABLE ".center(90, "="))
        print("=" * 90)
        print(metrics_df.to_string(index=False))
        print("=" * 90 + "\n")

        # Log individual area breakdowns under nested child runs using the real area name
        for _, row in metrics_df.iterrows():
            # Extract actual area name, fallback safely if column naming varies
            raw_area_name = row.get('area_name') or row.get('model_area_id')
            
            if pd.notna(raw_area_name):
                area_name = str(raw_area_name).strip()
            else:
                area_name = "unknown_area"
                
            clean_area = area_name.replace(" ", "_").lower()

            # Create a nested child run for each specific area
            with mlflow.start_run(run_name=f"model_{clean_area}", nested=True):
                mlflow.set_tag("area_name", area_name)
                
                for col in metrics_df.columns:
                    if col not in ['area_name', 'model_area_id']:
                        val = row[col]
                        if pd.notna(val):
                            if isinstance(val, (int, float)):
                                mlflow.log_metric(f"{col}", float(val))
                            else:
                                mlflow.log_param(f"{col}", str(val))

        # Save metrics table as an MLflow artifact on the parent run
        if not str(read_path).startswith('s3://'):
            mlflow.log_artifact(str(read_path), artifact_path="modeling_metrics")
            
        logger.info("Area-wise metrics logged successfully to MLflow with true area names.")
    else:
        logger.warning(f"Metrics file not found at {read_path}")


if __name__ == '__main__':
    stage = Path(__file__).resolve().parent.name
    if stage in SCRIPT_MAP:
        config = load_config()
        
        # Setup MLflow Tracking
        mlflow_cfg = config.get('mlflow', {})
        if mlflow_cfg.get('tracking_uri'):
            mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
        
        # Point everything to one single unified experiment name
        experiment_name = mlflow_cfg.get('experiment_name', 'truestates-ml-ops')
        mlflow.set_experiment(experiment_name)
        
        script_path = ROOT / SCRIPT_MAP[stage]
        
        # Open ONE single parent run for the entire stage execution
        with mlflow.start_run(run_name=f"stage_{stage}") as parent_run:
            mlflow.log_param("stage", stage)
            print(f"--- Running stage: {stage} via {script_path.name} (Parent Run ID: {parent_run.info.run_id}) ---")
            
            # Execute underlying multi-model regression script
            env = os.environ.copy()
            env["MLFLOW_RUN_ID"] = parent_run.info.run_id
            subprocess.run(['python', str(script_path)], check=True, env=env)
            
            # Print table and log true area-wise data as nested child runs under this parent run
            try:
                execute_modeling_tracking(config)
            except Exception as e:
                logger.error(f"Error logging modeling metadata to MLflow: {e}")
    else:
        print(f"Unknown stage directory: '{stage}'")
