"""
Stage G: Forecasting_news (Chronos + Macro-News adjustment)
Wraps forecasting_engine_chronos_news.execute_pipeline_entry()
DVC: G1 - alternate forecast CSV (adjusted_macro_forecast.csv) tracked via dvc.yaml
MLflow: G2 - logs alternate run tracking (params + adjustment metadata)
"""
import os
import sys
import time
import logging
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import init_mlflow, log_config_params

logger = logging.getLogger(__name__)


def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_forecasting_news"])
    with mlflow.start_run(run_name="forecasting_chronos_news"):
        log_config_params(config.get("macro_news_settings", {}), prefix="news.")

        t0 = time.time()
        import forecasting_engine_chronos_news as news_module
        result = news_module.execute_pipeline_entry(config)
        duration = time.time() - t0
        mlflow.log_metric("forecasting_news_duration_sec", duration)

        out_path = os.path.join(config["paths"]["base_dir"], config["paths"]["adjusted_forecast_output"])
        if os.path.exists(out_path):
            mlflow.log_artifact(out_path)

        context_path = os.path.join(config["paths"]["base_dir"], config["paths"]["news_context_file"])
        if os.path.exists(context_path):
            mlflow.log_artifact(context_path)

        logger.info(f"Forecasting_news complete in {duration:.2f}s")
        return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
