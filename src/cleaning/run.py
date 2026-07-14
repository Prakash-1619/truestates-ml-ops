"""
Stage C: Cleaning
Wraps transactions_data_main_parquet_yaml.run_transaction_processing()
MLflow: C1 - logs cleaning params (filters, exclusions, row counts before/after)
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
    init_mlflow(config["mlflow"]["experiment_cleaning"])
    with mlflow.start_run(run_name="cleaning"):
        log_config_params(config.get("transaction_settings", {}), prefix="cleaning.")

        raw_path = os.path.join(config["paths"]["base_dir"], config["paths"]["transactions_input"])
        rows_before = None
        if os.path.exists(raw_path):
            rows_before = pd.read_parquet(raw_path, columns=None).shape[0]
            mlflow.log_metric("cleaning_rows_before", rows_before)

        t0 = time.time()
        import transactions_data_main_parquet_yaml as cleaning_module
        result = cleaning_module.run_transaction_processing(config)
        duration = time.time() - t0

        out_path = os.path.join(config["paths"]["base_dir"], config["paths"]["transactions_output"])
        if os.path.exists(out_path):
            rows_after = pd.read_parquet(out_path).shape[0]
            mlflow.log_metric("cleaning_rows_after", rows_after)
            if rows_before:
                mlflow.log_metric("cleaning_rows_dropped_pct", 1 - rows_after / rows_before)

        mlflow.log_metric("cleaning_duration_sec", duration)
        logger.info(f"Cleaning complete in {duration:.2f}s")
        return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
