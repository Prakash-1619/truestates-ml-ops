"""
Main orchestrator for the TruEstates ML-Ops pipeline.
Runs: Ingestion -> Cleaning -> Merging -> Modeling -> Forecasting -> Forecasting_news
Each stage wrapper (src/<stage>/run.py) handles its own MLflow run + DVC-tracked outputs.
"""
# --- DagsHub Authentication ---
import os
import sys
import time
import logging
import yaml
import warnings
import subprocess
import s3fs # Required for s3:// protocol streaming

# --- DagsHub Authentication ---
token = "8df26f9f871b7249cc698426d87853f4ea3d8655"
os.environ["AWS_ACCESS_KEY_ID"] = token
os.environ["AWS_SECRET_ACCESS_KEY"] = token

# FIX 1: PyArrow requires the standard AWS_ENDPOINT_URL (without _S3)
os.environ["AWS_ENDPOINT_URL"] = "https://dagshub.com/poojariprakash88/truestates-ml-ops.s3"

# FIX 2: PyArrow requires a default region, otherwise it can crash during resolution
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# FIX 3: Forces PyArrow to format the S3 URL correctly for DagsHub's custom servers
os.environ["AWS_S3_FORCE_PATH_STYLE"] = "true"
warnings.filterwarnings("ignore")

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PIPELINE_DIR, "config.yaml")

# Timestamped log filename
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
        
    # --- BULLETPROOF S3 PATH INJECTOR ---
    # This automatically intercepts the config and forces the s3:// protocol 
    # onto all paths so Pandas knows to stream from DagsHub.
    for key, value in config['paths'].items():
        if isinstance(value, str):
            # If the string already has the bucket name but is missing s3://
            if value.startswith("truestates-ml-ops/"):
                config['paths'][key] = "s3://" + value
                
            # If it is just a plain filename, we map it to the correct S3 folder!
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

    # --- Auto-Sync Log to DagsHub via DVC ---
    logger.info("Syncing logs to DagsHub...")
    subprocess.run(["dvc", "add", log_filename], check=True)
    subprocess.run(["dvc", "push", f"{log_filename}.dvc", "-r", "origin"], check=True)
    
    logger.info(f"PIPELINE COMPLETE. Log saved as {log_filename}")

if __name__ == "__main__":
    run_full_dubai_pipeline()
