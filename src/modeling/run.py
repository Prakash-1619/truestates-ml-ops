"""
Stage E: Modeling
Wraps regression_modeling_log_parquet_yaml_multi.run_model_training()
DVC: E1 - train/test split artifacts + trained model files (models/*.pkl) tracked via dvc.yaml
MLflow: E2 - logs params, metrics (all_area_metrics.csv), and the model itself.
Also runs MODEL DRIFT check against the last run in this experiment.
"""
import os
import sys
import time
import logging
import pandas as pd
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import init_mlflow, log_config_params
from src.utils.drift_utils import check_model_drift

logger = logging.getLogger(__name__)


def run(config: dict):
    init_mlflow(config["mlflow"]["experiment_modeling"])
    with mlflow.start_run(run_name="modeling") as run_ctx:
        log_config_params(config.get("model_params", {}), prefix="model.")
        log_config_params(config.get("training_logic", {}), prefix="train.")

        t0 = time.time()
        import regression_modeling_log_parquet_yaml_multi as modeling_module
        result = modeling_module.run_model_training() #config
        duration = time.time() - t0
        mlflow.log_metric("modeling_duration_sec", duration)

        metrics_path = os.path.join(config["paths"]["base_dir"], config["paths"]["metrics_file"])
        agg_metrics = {}
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            mlflow.log_artifact(metrics_path)
            for col in metrics_df.select_dtypes("number").columns:
                agg_metrics[col] = float(metrics_df[col].mean())
                mlflow.log_metric(f"avg_{col}", agg_metrics[col])

        models_dir = os.path.join(config["paths"]["base_dir"], config["paths"]["models_dir"])
        if os.path.isdir(models_dir):
            mlflow.log_artifacts(models_dir, artifact_path="models")

        # --- Model drift check ---
        metric_key = config.get("drift", {}).get("model_metric_key", "r2")
        avg_key = f"avg_{metric_key}"
        if avg_key in agg_metrics:
            check_model_drift({metric_key: agg_metrics[avg_key]},
                               experiment_name=config["mlflow"]["experiment_modeling"],
                               metric_key=metric_key)

        logger.info(f"Modeling complete in {duration:.2f}s")
        return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
