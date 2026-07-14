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

warnings.filterwarnings("ignore")

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PIPELINE_DIR, "config.yaml")

for sub in ["src/ingestion", "src/cleaning", "src/merging", "src/modeling",
            "src/forecasting", "src/forecasting_news", "src/utils", "."]:
    p = os.path.join(PIPELINE_DIR, sub)
    if p not in sys.path:
        sys.path.append(p)

log_filename = "pipeline_run.log"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"CRITICAL: config.yaml not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


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

    logger.info("=" * 60)
    logger.info("TRUESTATES ML-OPS: FULL PIPELINE STARTING")
    logger.info("=" * 60)

    steps_to_run = steps_to_run or [
        "Ingestion",
        "Cleaning",
        "Merging",
        "Modeling",
        "Forecasting",
        "Forecasting_news",
    ]

    for i, step_name in enumerate(steps_to_run, 1):
        step_start = time.time()
        logger.info(f"--- [STEP {i}/{len(steps_to_run)}]: {step_name} ---")
        try:
            run_stage(step_name, config)
            duration = time.time() - step_start
            logger.info(f"{step_name} finished in {duration:.2f}s")
        except Exception as e:
            logger.error(f"ERROR in {step_name}: {e}", exc_info=True)
            raise

    total_duration = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {total_duration/60:.2f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]
    run_full_dubai_pipeline(steps_to_run=args if args else None)
