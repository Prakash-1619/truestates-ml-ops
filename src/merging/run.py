"""
Stage D: Merging
Wraps transactions_data_preparation_mode_parquet_yaml.run_merging_pipeline()
DVC: D1 - the merged dataset (latest_combined_data.parquet) is tracked via
     `dvc add data/processed/latest_combined_data.parquet` in dvc.yaml (see below).
MLflow: D2 - logs merge stats (row/col counts, null %, outlier counts) + data drift check.
"""
import os
import sys
import time
import logging
import pandas as pd
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import init_mlflow, log_config_params
from src.utils.drift_utils import check_data_drift

logger = logging.getLogger(__name__)


def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_merging"])
    with mlflow.start_run(run_name="merging"):
        log_config_params(config.get("merging_params", {}), prefix="merging.")

        t0 = time.time()
        import transactions_data_preparation_mode_parquet_yaml as merging_module
        result = merging_module.run_merging_pipeline(config)
        duration = time.time() - t0

        out_path = os.path.join(config["paths"]["base_dir"], config["paths"]["merging_output"])
        current_df = None
        if os.path.exists(out_path):
            current_df = pd.read_parquet(out_path)
            n_rows, n_cols = current_df.shape
            mlflow.log_metric("merge_rows", n_rows)
            mlflow.log_metric("merge_cols", n_cols)
            null_pct = float(current_df.isna().mean().mean())
            mlflow.log_metric("merge_avg_null_pct", null_pct)

        outliers_path = os.path.join(config["paths"]["base_dir"], config["paths"]["outliers_metadata"])
        if os.path.exists(outliers_path):
            mlflow.log_artifact(outliers_path)

        mlflow.log_metric("merge_duration_sec", duration)

        # --- Data drift check: compare against previous DVC-tracked snapshot ---
        prev_path = out_path + ".prev"
        if current_df is not None and os.path.exists(prev_path):
            baseline_df = pd.read_parquet(prev_path)
            numeric_cols = config.get("drift", {}).get("numeric_cols_for_data_drift", [])
            check_data_drift(baseline_df, current_df, numeric_cols,
                              artifact_name="merge_data_drift_report.json")
        else:
            logger.info("No previous snapshot found for drift comparison (first run).")

        if current_df is not None:
            current_df.to_parquet(prev_path, index=False)  # snapshot for next run's drift check

        logger.info(f"Merging complete in {duration:.2f}s")
        return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
