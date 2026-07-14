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

        out_path = os.path.join(config["paths"]["base_dir"], config["paths"]["chronos_output"])
        if os.path.exists(out_path):
            mlflow.log_artifact(out_path)

        backtest_path = os.path.join(config["paths"]["base_dir"], config["paths"]["chronos_backtest_output"])
        if os.path.exists(backtest_path):
            bt_df = pd.read_csv(backtest_path)
            mlflow.log_artifact(backtest_path)
            for col in bt_df.select_dtypes("number").columns:
                mlflow.log_metric(f"backtest_avg_{col}", float(bt_df[col].mean()))

        logger.info(f"Forecasting complete in {duration:.2f}s")
        return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
