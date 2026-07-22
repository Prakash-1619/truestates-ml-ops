import importlib
import logging
import sys
import time
from pathlib import Path
import warnings
import yaml

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'config.yaml'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(ROOT / 'pipeline_run.log', mode='w'), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODULE_MAP = {
    'Ingestion': {'module': 'micro_data_preparation_yaml', 'func': 'run_ingestion'},
    'Cleaning': {'module': 'transactions_data_main_parquet_yaml', 'func': 'run_transaction_processing'},
    'Merging': {'module': 'transactions_data_preparation_mode_parquet_yaml', 'func': 'run_merging_pipeline'},
    'Modeling': {'module': 'regression_modeling_log_parquet_yaml_multi', 'func': 'run_model_training'},
    'Forecasting': {'module': 'forecasting_engine_chronos', 'func': 'execute_pipeline_entry'},
    'Forecasting_news': {'module': 'forecasting_engine_chronos_news', 'func': 'execute_pipeline_entry'},
}

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_step(step_name, config):
    info = MODULE_MAP[step_name]
    mod = importlib.import_module(info['module'])
    importlib.reload(mod)
    for name in [info['func'], 'main', 'run', 'run_pipeline', 'execute_pipeline']:
        if hasattr(mod, name) and callable(getattr(mod, name)):
            func = getattr(mod, name)
            logger.info("Using function '%s' from module '%s'", name, info['module'])
            try:
                return func(config)
            except TypeError:
                return func()
    raise AttributeError(f"No callable entrypoint found in module {info['module']}")

def ensure_directories(config):
    required = [
        'base_dir', 'data_dir', 'raw_dir', 'processed_dir',
        'utils_dir', 'model_requirements_dir', 'models_dir', 'columns_dir'
    ]
    for key in required:
        if key not in config['paths']:
            raise KeyError(f"Missing key in config.yaml: 'paths.{key}'")
        Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)

def run_full_dubai_pipeline(steps_to_run=None):
    config = load_config()
    ensure_directories(config)
    steps_to_run = steps_to_run or [
        'Ingestion',
        'Cleaning',
        'Merging',
        'Modeling',
        'Forecasting',
        'Forecasting_news'
    ]
    start = time.time()
    logger.info('=' * 60)
    logger.info('TRUESTATES ML OPS PIPELINE STARTING')
    logger.info('=' * 60)
    for idx, step_name in enumerate(steps_to_run, 1):
        step_start = time.time()
        logger.info('--- [STEP %s / %s]: %s ---', idx, len(steps_to_run), step_name)
        run_step(step_name, config)
        logger.info('Completed %s in %.2f s', step_name, time.time() - step_start)
    logger.info('Pipeline complete in %.2f minutes', (time.time() - start) / 60)

if __name__ == '__main__':
    run_full_dubai_pipeline()
