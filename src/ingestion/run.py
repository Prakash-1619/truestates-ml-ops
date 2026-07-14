"""
Stage B: Ingestion
Wraps micro_data_preparation_yaml.run_ingestion()
MLflow: B1 - logs ingestion metadata (row counts, column counts, config params)
"""
import os
import sys
import time
import logging
import pandas as pd
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import init_mlflow, log_config_params

logger = logging.getLogger(__name__)


def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_ingestion"])
    with mlflow.start_run(run_name="ingestion"):
        log_config_params(config.get("ingestion_columns", {}), prefix="ingestion.")
        log_config_params(config.get("ingestion_processing", {}), prefix="ingestion.")

        t0 = time.time()
        import micro_data_preparation_yaml as ingestion_module
        result = ingestion_module.run_ingestion() #config
        duration = time.time() - t0

        out_path = os.path.join(config["paths"]["base_dir"], config["paths"]["ingestion_output"])
        n_rows, n_cols = None, None
        if os.path.exists(out_path):
            df = pd.read_csv(out_path, nrows=None)
            n_rows, n_cols = df.shape
            mlflow.log_metric("ingestion_rows", n_rows)
            mlflow.log_metric("ingestion_cols", n_cols)

        mlflow.log_metric("ingestion_duration_sec", duration)
        mlflow.log_param("ingestion_output_file", config["paths"]["ingestion_output"])
        logger.info(f"Ingestion complete: rows={n_rows}, cols={n_cols}, duration={duration:.2f}s")
        return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
