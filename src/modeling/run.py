"""
Stage E: Modeling
Wraps regression_modeling_log_parquet_yaml_multi.run_model_training()
"""
import os
import sys
import time
import logging
import pandas as pd
import mlflow
from dagshub import get_repo_bucket_client

# 1. Initialize DagsHub S3 Client
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import init_mlflow, log_config_params
from src.utils.drift_utils import check_model_drift

logger = logging.getLogger(__name__)

def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_modeling"])
    with mlflow.start_run(run_name="modeling") as run_ctx:
        # Log global parameters
        log_config_params(config.get("model_params", {}), prefix="model.")
        log_config_params(config.get("training_logic", {}), prefix="train.")
        
        t0 = time.time()
        import regression_modeling_log_parquet_yaml_multi as modeling_module
        result = modeling_module.run_model_training()
        
        duration = time.time() - t0
        mlflow.log_metric("modeling_duration_sec", duration)

        # 2. Construct S3 paths natively
        base = config["paths"]["base_dir"].replace("s3://", "")
        metrics_path = f"{base}/{config['paths'].get('metrics_file', 'processed/all_area_metrics.csv')}"
        params_path = f"{base}/{config['paths'].get('params_file', 'model_requirements/all_param_logs.csv')}"
        
        agg_metrics = {}
        
        # 3. Log ALL AREA METRICS to MLflow
        if fs.exists(metrics_path):
            with fs.open(metrics_path, "rb") as f:
                metrics_df = pd.read_csv(f)
            
            metrics_df.to_csv("all_area_metrics.csv", index=False)
            mlflow.log_artifact("all_area_metrics.csv") # Uploads to dashboard
            
            for col in metrics_df.select_dtypes("number").columns:
                agg_metrics[col] = float(metrics_df[col].mean())
                mlflow.log_metric(f"avg_{col}", agg_metrics[col])
        else:
            logger.warning(f"⚠️ Metrics file not found in S3 at {metrics_path}")

        # 4. Log ALL AREA PARAMETERS to MLflow
        if fs.exists(params_path):
            with fs.open(params_path, "rb") as f:
                params_df = pd.read_csv(f)
            
            params_df.to_csv("all_param_logs.csv", index=False)
            mlflow.log_artifact("all_param_logs.csv") # Uploads specific area params to dashboard
        
        # Note: We skip mlflow.log_artifacts(models_dir) because MLflow cannot 
        # upload from an S3 source, and your 27 models are already safely stored 
        # directly in your DagsHub S3 bucket!

        # --- Model drift check ---
        metric_key = config.get("drift", {}).get("model_metric_key", "r2")
        avg_key = f"avg_{metric_key}"
        if avg_key in agg_metrics:
            check_model_drift({metric_key: agg_metrics[avg_key]}, experiment_name=config["mlflow"]["experiment_modeling"], metric_key=metric_key)
            
        logger.info(f"Modeling complete in {duration:.2f}s")
        return result

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
