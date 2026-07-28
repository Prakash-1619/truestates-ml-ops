import importlib
import logging
import sys
import time
from pathlib import Path
import warnings
import yaml
import mlflow
import os

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'config.yaml'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / 'pipeline_run.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)



# Set Cloudflare R2 Environment Variables for S3FS & Pandas
os.environ["AWS_ENDPOINT_URL"] = "https://ef8eef61229ee8854b4237f6949e50d8.r2.cloudflarestorage.com/truestates-re-analytics"
os.environ["AWS_ACCESS_KEY_ID"] = "c198c85bd01da0931eae24009fb2100b"
os.environ["AWS_SECRET_ACCESS_KEY"] = "826187ffaee4742816f65ca4ebe149902db75ac52dbb81606bb34fe8bae4a57c"

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

def execute_stage_tracking(step_name, config):
    """
    Safely loads and executes the custom MLflow tracking functions 
    for metrics parsing and summary tables.
    """
    try:
        if step_name == 'Modeling':
            from src.modeling.run import execute_modeling_tracking
            execute_modeling_tracking(config)
        elif step_name == 'Forecasting':
            from src.forecasting.run import execute_forecasting_tracking
            execute_forecasting_tracking(config)
        elif step_name == 'Forecasting_news':
            from src.forecasting_news.run import execute_forecasting_news_tracking
            execute_forecasting_news_tracking(config)
        else:
            logger.info(f"No custom MLflow metrics tracking function defined for {step_name} yet.")
    except ImportError as e:
        logger.warning(f"MLflow tracking helper not found for {step_name}. Skipping trace. Error: {e}")
    except Exception as e:
        logger.error(f"Error during MLflow tracking for {step_name}: {e}")

def ensure_directories(config):
    required = ['base_dir', 'data_dir', 'raw_dir', 'processed_dir', 'utils_dir', 'model_requirements_dir', 'models_dir', 'columns_dir']
    for key in required:
        if key not in config['paths']:
            raise KeyError(f"Missing key in config.yaml: 'paths.{key}'")
        
        path_str = config['paths'][key]
        
        # Only attempt to create local folders if path exists and is NOT an S3 URI
        if path_str and not path_str.startswith("s3://"):
            Path(path_str).mkdir(parents=True, exist_ok=True)
            
def run_full_dubai_pipeline(steps_to_run=None):
    config = load_config()
    ensure_directories(config)
    
    mlflow_cfg = config.get('mlflow', {})
    if mlflow_cfg.get('tracking_uri'):
        mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg.get('experiment_name', 'truestates-ml-ops'))

    steps_to_run = steps_to_run or ['Ingestion', ] #' , 'Merging','Cleaning', 'Merging','Modeling','Forecasting','Forecasting_news'
    start = time.time()
    logger.info('=' * 60)
    logger.info('TRUESTATES ML OPS PIPELINE STARTING')
    logger.info('=' * 60)

    with mlflow.start_run(run_name="full_pipeline_run"):
        for idx, step_name in enumerate(steps_to_run, 1):
            step_start = time.time()
            logger.info('--- [STEP %s / %s]: %s ---', idx, len(steps_to_run), step_name)
            
            with mlflow.start_run(run_name=f"stage_{step_name.lower()}", nested=True):
                mlflow.log_param("stage", step_name)
                
                # 1. Execute the heavy machine learning script
                run_step(step_name, config)
                
                # 2. Execute the MLflow tracking helper to parse metrics and push artifacts
                execute_stage_tracking(step_name, config)

            logger.info('Completed %s in %.2f s', step_name, time.time() - step_start)
            
    logger.info('Pipeline complete in %.2f minutes', (time.time() - start) / 60)

if __name__ == '__main__':
    run_full_dubai_pipeline()