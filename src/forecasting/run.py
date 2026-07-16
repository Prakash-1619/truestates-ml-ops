"""
Stage F: Forecasting (Chronos)
Wraps forecasting_engine_chronos.execute_pipeline_entry()
DVC: F1 - forecast CSV (final_chronos_forecasts.csv) tracked via dvc.yaml
MLflow: F2 - logs prediction artifact + backtest metrics
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

logger = logging.getLogger(__name__)

def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_forecasting"])
    with mlflow.start_run(run_name="forecasting_chronos"):
        log_config_params(config.get("forecast_settings", {}), prefix="forecast.")
        log_config_params(config.get("forecasting_settings", {}), prefix="forecast.")
        
        t0 = time.time()
        import forecasting_engine_chronos as forecasting_module
        result = forecasting_module.execute_pipeline_entry(config)
        duration = time.time() - t0
        mlflow.log_metric("forecasting_duration_sec", duration)

        # 2. Form S3 paths natively
        base = config["paths"]["base_dir"].replace("s3://", "")
        out_path = f"{base}/{config['paths']['chronos_output']}"
        backtest_path = f"{base}/{config['paths']['chronos_backtest_output']}"

        # 3. Log Forecast CSV Artifact
        if fs.exists(out_path):
            with fs.open(out_path, "rb") as f:
                forecast_df = pd.read_csv(f)
            # Save temporarily to upload to MLflow
            forecast_df.to_csv("final_chronos_forecasts.csv", index=False)
            mlflow.log_artifact("final_chronos_forecasts.csv")
        else:
            logger.warning(f"⚠️ Forecast file not found in S3 at {out_path}")

        # 4. Log Area-Wise Backtest Metrics
        if fs.exists(backtest_path):
            with fs.open(backtest_path, "rb") as f:
                bt_df = pd.read_csv(f)
            
            # Save temporarily to upload to MLflow
            bt_df.to_csv("chronos_backtest_metrics.csv", index=False)
            mlflow.log_artifact("chronos_backtest_metrics.csv")
            
            numeric_cols = bt_df.select_dtypes("number").columns
            
            # A. Log overall averages
            for col in numeric_cols:
                if col != 'model_area_id': # Skip averaging the ID column
                    mlflow.log_metric(f"backtest_avg_{col}", float(bt_df[col].mean()))
            
            # B. Log specific AREA-WISE metrics to the MLflow dashboard
            if 'model_area_id' in bt_df.columns:
                for _, row in bt_df.iterrows():
                    area_id = int(row['model_area_id'])
                    for col in numeric_cols:
                        if col != 'model_area_id':
                            # This will create metrics like: area_14_backtest_mape
                            mlflow.log_metric(f"area_{area_id}_backtest_{col}", float(row[col]))
                            
        else:
            logger.warning(f"⚠️ Backtest metrics not found in S3 at {backtest_path}")

        logger.info(f"Forecasting complete in {duration:.2f}s")
        return result

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
