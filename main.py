import os
import sys
import time
import logging
import yaml
import warnings
import subprocess
import pandas as pd
import s3fs

# --- DagsHub Authentication ---
token = "8df26f9f871b7249cc698426d87853f4ea3d8655"
os.environ["AWS_ACCESS_KEY_ID"] = token
os.environ["AWS_SECRET_ACCESS_KEY"] = token
os.environ["AWS_ENDPOINT_URL"] = "https://dagshub.com/poojariprakash88/truestates-ml-ops.s3"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_S3_FORCE_PATH_STYLE"] = "true"

# =====================================================================
# PANDAS S3 HIJACKER (Monkey Patch)
# =====================================================================

fs = s3fs.S3FileSystem(
    key=token,
    secret=token,
    client_kwargs={"endpoint_url": "https://dagshub.com/poojariprakash88/truestates-ml-ops.s3"},
)

# 1. Intercept read_parquet
_orig_read_parquet = pd.read_parquet

def _safe_read_parquet(path, *args, **kwargs):
    if isinstance(path, str) and path.startswith("s3://"):
        clean_path = path.replace("s3://", "")
        with fs.open(clean_path, "rb") as f:
            return _orig_read_parquet(f, *args, **kwargs)
    return _orig_read_parquet(path, *args, **kwargs)

pd.read_parquet = _safe_read_parquet

# 2. Intercept read_csv
_orig_read_csv = pd.read_csv

def _safe_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, str) and filepath_or_buffer.startswith("s3://"):
        clean_path = filepath_or_buffer.replace("s3://", "")
        with fs.open(clean_path, "rb") as f:
            return _orig_read_csv(f, *args, **kwargs)
    return _orig_read_csv(filepath_or_buffer, *args, **kwargs)

pd.read_csv = _safe_read_csv

# 3. Intercept to_parquet
_orig_to_parquet = pd.DataFrame.to_parquet

def _safe_to_parquet(self, path=None, *args, **kwargs):
    if isinstance(path, str) and path.startswith("s3://"):
        clean_path = path.replace("s3://", "")
        with fs.open(clean_path, "wb") as f:
            return _orig_to_parquet(self, f, *args, **kwargs)
    return _orig_to_parquet(self, path, *args, **kwargs)

pd.DataFrame.to_parquet = _safe_to_parquet

# 4. Intercept to_csv
_orig_to_csv = pd.DataFrame.to_csv

def _safe_to_csv(self, path_or_buf=None, *args, **kwargs):
    if isinstance(path_or_buf, str) and path_or_buf.startswith("s3://"):
        clean_path = path_or_buf.replace("s3://", "")
        with fs.open(clean_path, "wb") as f:
            return _orig_to_csv(self, f, *args, **kwargs)
    return _orig_to_csv(self, path_or_buf, *args, **kwargs)

pd.DataFrame.to_csv = _safe_to_csv

# =====================================================================

warnings.filterwarnings("ignore")

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PIPELINE_DIR, "config.yaml")

log_filename = f"pipeline_run_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config():
    """
    Load config.yaml and rewrite `paths` to s3:// locations.

    Raw files -> s3://.../data/raw/
    Processed outputs -> s3://.../data/processed/
    Model artifacts -> s3://.../data/model_requirements/
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    bucket = "truestates-ml-ops"
    base_raw = f"s3://{bucket}/data/raw"
    base_processed = f"s3://{bucket}/data/processed"
    base_models = f"s3://{bucket}/data/model_requirements"

    processed_keys = [
        "ingestion_output",
        "transactions_output",
        "merging_output",
        "outliers_metadata",
        "metrics_file",
        "forecast_metadata",
        "chronos_backtest_output",
    ]

    model_keys = [
        "models_dir",
        "columns_dir",
        "ranges_file",
        "historic_output",
        "growth_output",
        "chronos_output",
        "chronos_historic_output",
        "adjusted_forecast_output",
    ]

    for key, value in config["paths"].items():
        if not isinstance(value, str):
            continue

        # Already a full s3 path -> leave as is
        if value.startswith("s3://"):
            continue

        # Utility paths (JSON etc.) -> keep relative
        if value.startswith("utils/"):
            continue

        filename = value.split("/")[-1]

        if key in processed_keys:
            config["paths"][key] = f"{base_processed}/{filename}"
        elif key in model_keys:
            # For directories, keep as 'dir'; for files, treat as file
            if filename.endswith((".parquet", ".csv")):
                config["paths"][key] = f"{base_models}/{filename}"
            else:
                config["paths"][key] = f"{base_models}/{filename}"
        elif filename.endswith((".parquet", ".csv", ".xlsx")):
            # Raw data files
            config["paths"][key] = f"{base_raw}/{filename}"
        else:
            # Fallback: keep original
            config["paths"][key] = value

    return config


# ---------------------------------------------------------------------
# Stage wrappers calling your existing scripts
# ---------------------------------------------------------------------

def run_ingestion_stage(config):
    """
    Ingestion: projects + developers + buildings + units -> latest_pdbu_df.csv
    Implemented in micro_data_preparation_yaml.py as run_ingestion(config).
    """
    from micro_data_preparation_yaml import run_ingestion
    logger.info("Running INGESTION stage...")
    return run_ingestion(config)


def run_cleaning_merging_stage(config):
    """
    Cleaning & Merging:
    - Clean raw transactions (parquet)
    - Merge with micro data on transubpkey
    Implemented in transactions_data_main_parquet_yaml.py as run_merging_pipeline().
    """
    from transactions_data_main_parquet_yaml import run_merging_pipeline
    logger.info("Running CLEANING + MERGING stage...")
    return run_merging_pipeline()


def run_modeling_stage(config):
    """
    Modeling:
    - Train XGBoost / CatBoost / RF models on merged parquet
    Implemented in regression_modeling_log_parquet_yaml_multi.py as run_model_training().
    """
    from regression_modeling_log_parquet_yaml_multi import run_model_training
    logger.info("Running MODELING stage...")
    return run_model_training()


def run_forecasting_stage(config):
    """
    Forecasting:
    - Chronos-based forecasting on monthly aggregated data
    Implemented in forecasting_engine_chronos.py as execute_pipeline_entry(config).
    """
    from forecasting_engine_chronos import execute_pipeline_entry
    logger.info("Running FORECASTING stage...")
    return execute_pipeline_entry(config)


def run_full_dubai_pipeline(steps_to_run=None):
    config = load_config()
    pipeline_start = time.time()
    logger.info("TRUESTATES ML-OPS: PIPELINE STARTING")

    steps = steps_to_run or [
        "Ingestion",
        "Cleaning_Merging",
        "Modeling",
        "Forecasting",
    ]

    for i, step_name in enumerate(steps, 1):
        logger.info(f"--- [STEP {i}/{len(steps)}]: {step_name} ---")
        if step_name == "Ingestion":
            run_ingestion_stage(config)
        elif step_name == "Cleaning_Merging":
            run_cleaning_merging_stage(config)
        elif step_name == "Modeling":
            run_modeling_stage(config)
        elif step_name == "Forecasting":
            run_forecasting_stage(config)
        else:
            logger.warning(f"Unknown step {step_name}, skipping.")

    logger.info("Syncing logs to DagsHub...")
    subprocess.run(["dvc", "add", log_filename], check=True)
    subprocess.run(["dvc", "push", f"{log_filename}.dvc", "-r", "origin"], check=True)

    elapsed = time.time() - pipeline_start
    logger.info(f"PIPELINE COMPLETE in {elapsed:.2f}s. Log saved as {log_filename}")


if __name__ == "__main__":
    run_full_dubai_pipeline()
