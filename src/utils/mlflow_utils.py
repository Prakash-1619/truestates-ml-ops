from pathlib import Path
import mlflow
import yaml


def load_project_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_mlflow(config=None, config_path='config.yaml'):
    config = config or load_project_config(config_path)
    tracking_uri = config['mlflow']['tracking_uri']
    experiment_name = config['mlflow']['experiment_name']
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri, experiment_name


def ensure_project_dirs(config):
    for key in ['data_dir', 'raw_dir', 'processed_dir', 'utils_dir', 'model_requirements_dir', 'models_dir', 'columns_dir']:
        Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)
