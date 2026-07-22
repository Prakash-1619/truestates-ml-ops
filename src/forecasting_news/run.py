from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_MAP = {
    'ingestion': 'micro_data_preparation_yaml.py',
    'cleaning': 'transactions_data_main_parquet_yaml.py',
    'merging': 'transactions_data_preparation_mode_parquet_yaml.py',
    'modeling': 'regression_modeling_log_parquet_yaml_multi.py',
    'forecasting': 'forecasting_engine_chronos.py',
    'forecasting_news': 'forecasting_engine_chronos_news.py',
}

if __name__ == '__main__':
    stage = Path(__file__).resolve().parent.name
    subprocess.run(['python', str(ROOT / SCRIPT_MAP[stage])], check=True)
