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
# PyArrow's native S3 client strips the "s3://" prefix and ignores
# our DagsHub configurations. To stop this, we manually open the connection
# using s3fs and hand Pandas a raw byte stream so it never sees the URL!

fs = s3fs.S3FileSystem(
    key=token,
    secret=token,
    client_kwargs={"endpoint_url": "https://dagshub.com/poojariprakash88/truestates-ml-ops.s3"}
)

# 1. Intercept read_parquet
_orig_read_parquet = pd.read_parquet
def _safe_read_parquet(path, *args, **kwargs):
    if isinstance(path, str) and path.startswith("s3://"):
        clean_path = path.replace("s3://", "")  # <-- THIS FIXES THE 404!
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
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    for key, value in config['paths'].items():
        if isinstance(value, str):
            if value.startswith("truestates-ml-ops/"):
                config['paths'][key] = "s3://" + value
            elif not value.startswith("s3://") and not value.startswith("utils/"):
                filename = value.split("/")[-1]
                
                # Route Processed Files
                if key in ["ingestion_output", "transactions_output", "merging_output", "outliers_metadata", "metrics_file", "forecast_metadata", "chronos_backtest_output"]:
                    config['paths'][key] = f"s3://truestates-ml-ops/data/processed/{filename}"
                # Route Model Requirements
                elif key in ["models_dir", "columns_dir", "ranges_file", "historic_output", "growth_output", "chronos_output", "chronos_historic_output", "adjusted_forecast_output"]:
                    config['paths'][key] = f"s3://truestates-ml-ops/data/model_requirements/{filename}"
                # Route Raw Inputs
                elif value.endswith((".parquet", ".csv", ".xlsx")):
                    config['paths'][key] = f"s3://truestates-ml-ops/data/raw/{filename}"
                    
    return config

STAGE_MODULES = {
    "Ingestion": "src.ingestion.run",
    "Cleaning": "src.cleaning.run",
    "Merging": "src.merging.run",
    "Modeling": "src.modeling.run",
    "Forecasting": "src.forecasting.run",
    "Forecasting_news": "src.forecasting_news.run",
}

def run_stage(stage_name, config):
    import importlib
    mod = importlib.import_module(STAGE_MODULES[stage_name])
    importlib.reload(mod)
    return mod.run(config)

def run_full_dubai_pipeline(steps_to_run=None):
    config = load_config()
    pipeline_start = time.time()
    logger.info("TRUESTATES ML-OPS: PIPELINE STARTING")

    steps = steps_to_run or list(STAGE_MODULES.keys())

    for i, step_name in enumerate(steps, 1):
        logger.info(f"--- [STEP {i}/{len(steps)}]: {step_name} ---")
        run_stage(step_name, config)

    logger.info("Syncing logs to DagsHub...")
    subprocess.run(["dvc", "add", log_filename], check=True)
    subprocess.run(["dvc", "push", f"{log_filename}.dvc", "-r", "origin"], check=True)
    
    logger.info(f"PIPELINE COMPLETE. Log saved as {log_filename}")

if __name__ == "__main__":
    run_full_dubai_pipeline()
