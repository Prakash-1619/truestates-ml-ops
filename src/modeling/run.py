"""
Stage E: Modeling - Unified Area-Wise Logging
Reads metrics and params from S3, merges them, and logs to MLflow.
"""
import os
import sys
import logging
import pandas as pd
import mlflow
from dagshub import get_repo_bucket_client

# 1. Initialize Client
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import init_mlflow, log_config_params
from src.utils.drift_utils import check_model_drift

logger = logging.getLogger(__name__)

def _safe_name(name):
    """Clean name for MLflow keys (remove spaces, special chars)."""
    return str(name).strip().replace(' ', '_').replace("'", "").replace("/", "_").replace("-", "_")

def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_modeling"])
    
    with mlflow.start_run(run_name="modeling_unified"):
        # Log global config
        log_config_params(config.get("model_params", {}), prefix="model.")
        
        # Run Training
        import regression_modeling_log_parquet_yaml_multi as modeling_module
        modeling_module.run_model_training()
        
        # 2. Paths
        base = config["paths"]["base_dir"].replace("s3://", "")
        metrics_path = f"{base}/{config['paths'].get('metrics_file', 'processed/all_area_metrics.csv')}"
        params_path = f"{base}/{config['paths'].get('params_file', 'model_requirements/all_param_logs.csv')}"
        
        # 3. Read and Merge
        merged_df = None
        if fs.exists(metrics_path) and fs.exists(params_path):
            with fs.open(metrics_path, "rb") as f: metrics_df = pd.read_csv(f)
            with fs.open(params_path, "rb") as f: params_df = pd.read_csv(f)
            
            # Merge on area ID
            merged_df = pd.merge(metrics_df, params_df, on='model_area_id', how='inner')
            merged_df.to_csv("unified_area_report.csv", index=False)
            mlflow.log_artifact("unified_area_report.csv")
            logger.info("Merged metrics and params saved.")
        else:
            logger.error("Metrics or Params file missing in S3.")
            return

        # 4. Log Area-Wise Metrics to MLflow
        # This creates keys like: Area_14_DubaiMarina_test_r2
        for _, row in merged_df.iterrows():
            area_id = row['model_area_id']
            # Try to get area name, fallback to ID if missing
            area_name = row.get('area_name', 'NoName') 
            prefix = _safe_name(f"Area_{area_id}_{area_name}")
            
            for col in merged_df.columns:
                if col in ['model_area_id', 'area_name']: continue
                
                # If numeric, log as metric
                if pd.api.types.is_numeric_dtype(merged_df[col]):
                    val = float(row[col])
                    mlflow.log_metric(f"{prefix}_{col}", val)
                # If param/string, log as tag
                else:
                    mlflow.set_tag(f"{prefix}_{col}", str(row[col]))

        # 5. Drift Check
        if 'test_r2' in merged_df.columns:
            check_model_drift({'r2': float(merged_df['test_r2'].mean())}, 
                              experiment_name=config["mlflow"]["experiment_modeling"], 
                              metric_key='r2')

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
