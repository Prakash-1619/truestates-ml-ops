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

def execute_merging_tracking(config):
    """
    Extracts merging stage market snapshot metrics to track in MLflow:
    - Top sold areas and their volumes for the latest month in the dataset
    - Merged dataset dimensions
    """
    paths = config.get('paths', {})
    merged_file = paths.get('merged_file') or paths.get('combined_transactions_file') or "data/processed/transactions_merged.parquet"
    
    merged_path = ROOT / merged_file if not Path(merged_file).is_absolute() else Path(merged_file)

    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        
        # Dynamically identify month, area, and value columns
        month_col = next((col for col in df.columns if 'month' in col.lower() or 'date' in col.lower()), None)
        area_col = next((col for col in df.columns if 'area' in col.lower() or 'location' in col.lower()), None)
        value_col = next((col for col in df.columns if 'price' in col.lower() or 'amount' in col.lower() or 'volume' in col.lower()), None)
        
        if month_col and area_col:
            df[month_col] = pd.to_datetime(df[month_col], errors='coerce')
            latest_month = df[month_col].max()
            
            if pd.notna(latest_month):
                latest_df = df[df[month_col] == latest_month]
                
                if value_col:
                    top_areas = latest_df.groupby(area_col)[value_col].sum().sort_values(ascending=False).head(5).to_dict()
                else:
                    top_areas = latest_df[area_col].value_counts().head(5).to_dict()
                    
                for area, val in top_areas.items():
                    clean_area_name = str(area).replace(" ", "_").lower()
                    mlflow.log_metric(f"top_sold_volume_{clean_area_name}", float(val))
                    
                logger.info(f"Top sold areas logged for latest month ({latest_month.strftime('%Y-%m')}): {list(top_areas.keys())}")
        
        mlflow.log_metric("merged_dataset_rows", len(df))
        mlflow.log_param("merged_dataset_columns", len(df.columns))
        logger.info(f"Merging stage metrics logged: Rows={len(df)}, Columns={len(df.columns)}")

if __name__ == '__main__':
    # Automatically determine the stage based on the parent folder name
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
            
            # Execute underlying merging script
            subprocess.run(['python', str(script_path)], check=True)
            
            # Log market snapshot metrics to MLflow
            try:
                execute_merging_tracking(config)
            except Exception as e:
                logger.error(f"Error logging merging metadata to MLflow: {e}")
    else:
        logger.warning(f"Unknown stage directory: '{stage}'")