"""
Main orchestrator for the TruEstates ML-Ops pipeline.
Runs: Ingestion -> Cleaning -> Merging -> Modeling -> Forecasting -> Forecasting_news
Each stage wrapper (src/<stage>/run.py) handles its own MLflow run + DVC-tracked outputs.
"""
import os
import sys
import time
import logging
import yaml
import warnings
import subprocess
import s3fs # Required for s3:// protocol streaming
# Fix: Windows CP1252 can't encode MLflow emoji → UnicodeEncodeError.
if hasattr(sys.stdout, "buffer") and (not sys.stdout.encoding or sys.stdout.encoding.lower() != "utf-8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# --- DagsHub Authentication ---
# token = "8df26f9f871b7249cc698426d87853f4ea3d8655"
token = os.environ.get("DAGSHUB_TOKEN", "8df26f9f871b7249cc698426d87853f4ea3d8655")
os.environ["AWS_ACCESS_KEY_ID"] = token
os.environ["AWS_SECRET_ACCESS_KEY"] = token
os.environ["AWS_ENDPOINT_URL_S3"] = "https://dagshub.com/poojariprakash88/truestates-ml-ops.s3"

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
        return yaml.safe_load(f)

STAGE_MODULES = {
    #"Ingestion": "src.ingestion.run",
    #"Cleaning": "src.cleaning.run",
    #"Merging": "src.merging.run",
    #"Modeling": "src.modeling.run",
    #"Forecasting": "src.forecasting.run",
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
    try:
        dvc_cmd = [sys.executable, "-m", "dvc"]  # python -m dvc works even if dvc not on PATH
        subprocess.run(dvc_cmd + ["add", log_filename], check=True)
        subprocess.run(dvc_cmd + ["push", f"{log_filename}.dvc", "-r", "origin"], check=True)
        logger.info(f"PIPELINE COMPLETE. Log synced to DagsHub as {log_filename}")
    except Exception as e:
        logger.warning(f"DVC log sync skipped (non-critical): {e}")
        logger.info(f"PIPELINE COMPLETE. Log saved locally as {log_filename}")

if __name__ == "__main__":
    run_full_dubai_pipeline()
