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


def _to_local_path(path: str) -> str:
    if not path or not path.startswith('s3://'):
        return path
    remainder = path[len('s3://'):]
    if '/' in remainder:
        _, _, rel_path = remainder.partition('/')
        return os.path.join(os.getcwd(), rel_path)
    return os.path.join(os.getcwd(), remainder)


def _resolve_existing_path(path: str) -> str:
    if not path:
        return path
    try:
        if fs.exists(path):
            return path
    except Exception:
        pass
    local_path = _to_local_path(path)
    if os.path.exists(local_path):
        return local_path
    return local_path


def _open_existing_path(path: str, mode: str):
    if path.startswith('s3://'):
        try:
            return fs.open(path, mode)
        except Exception:
            local_path = _to_local_path(path)
            return open(local_path, mode)
    return open(path, mode)


# Numeric metric columns produced by regression_modeling_log_parquet_yaml_multi
AREA_METRIC_COLS = [
    'test_r2', 'test_mape', 'test_mae', 'test_rmse',
    'train_r2', 'train_mape', 'train_mae', 'train_rmse',
    'test_samples', 'train_samples',
    'median_actual_price', 'median_pred_price',
]

def _safe_area_key(area_name: str) -> str:
    """Convert area name to a safe MLflow metric key (no spaces/special chars)."""
    return str(area_name).strip().replace(' ', '_').replace("'", "").replace("/", "_").replace("-", "_")

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

        # 2. Resolve metrics and params paths, preferring local fallback when S3 is unavailable
        base = config["paths"]["base_dir"]
        metrics_path = f"{base}/{config['paths'].get('metrics_file', 'processed/all_area_metrics.csv')}"
        params_path = f"{base}/{config['paths'].get('params_file', 'model_requirements/all_param_logs.csv')}"
        metrics_path = _resolve_existing_path(metrics_path)
        params_path = _resolve_existing_path(params_path)

        agg_metrics = {}

        # 3. Log ALL AREA METRICS to MLflow
        if os.path.exists(metrics_path) or fs.exists(metrics_path):
            with _open_existing_path(metrics_path, "rb") as f:
                metrics_df = pd.read_csv(f)

            # Upload full CSV as artifact (visible in MLflow dashboard)
            metrics_df.to_csv("all_area_metrics.csv", index=False)
            mlflow.log_artifact("all_area_metrics.csv")

            numeric_cols = [c for c in AREA_METRIC_COLS if c in metrics_df.columns]
            area_col = 'area' if 'area' in metrics_df.columns else \
                       'area_name_en' if 'area_name_en' in metrics_df.columns else None

            # A. Log overall averages (used for drift detection)
            for col in numeric_cols:
                agg_metrics[col] = float(metrics_df[col].mean(skipna=True))
                mlflow.log_metric(f"avg_{col}", agg_metrics[col])

            # B. Log per-area metrics: e.g. test_r2__Business_Bay, test_mape__Mirdif
            if area_col:
                for _, row in metrics_df.iterrows():
                    area_key = _safe_area_key(row[area_col])
                    for col in numeric_cols:
                        if pd.notna(row.get(col)):
                            mlflow.log_metric(f"{col}__{area_key}", float(row[col]))
                    if 'best_algorithm' in metrics_df.columns:
                        try:
                            mlflow.log_param(f"best_algo__{area_key}", str(row.get('best_algorithm', '')))
                        except Exception:
                            pass  # param already logged or key too long

            logger.info(f"Per-area metrics logged for {len(metrics_df)} areas.")
        else:
            logger.warning(f"Metrics file not found in S3 at {metrics_path}")

        # 4. Log ALL AREA PARAMETERS to MLflow
        if os.path.exists(params_path) or fs.exists(params_path):
            with _open_existing_path(params_path, "rb") as f:
                params_df = pd.read_csv(f)

            params_df.to_csv("all_param_logs.csv", index=False)
            mlflow.log_artifact("all_param_logs.csv")

            # Log best CV metrics per area (cv_rank == 1 → best config per area)
            if 'cv_rank' in params_df.columns and 'area' in params_df.columns:
                best_cv = params_df[params_df['cv_rank'] == 1]
                for _, row in best_cv.iterrows():
                    area_key = _safe_area_key(row['area'])
                    for col in ['cv_mean_mape', 'cv_mean_r2']:
                        if col in row and pd.notna(row[col]):
                            mlflow.log_metric(f"{col}__{area_key}", float(row[col]))

        # Note: We skip mlflow.log_artifacts(models_dir) because MLflow cannot
        # upload from an S3 source, and your models are already safely stored
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
