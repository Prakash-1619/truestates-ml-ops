# TruEstates ML-Ops Pipeline

Pipeline structure (matches the required flowchart):

```
main.py
 ├── Ingestion        -> MLflow: ingestion metadata
 ├── Cleaning         -> MLflow: cleaning params
 ├── Merging          -> DVC: merged dataset | MLflow: merge stats + DATA DRIFT
 ├── Modeling         -> DVC: train/test split + model artifact | MLflow: params, metrics, model + MODEL DRIFT
 ├── Forecasting      -> DVC: forecast CSV | MLflow: prediction artifact
 └── Forecasting_news -> DVC: alternate forecast CSV | MLflow: alternate run tracking
```

## 1. Repo layout
```
.
├── main.py                    # orchestrator (was main_yaml.py)
├── config.yaml                # updated paths -> DagsHub raw bucket / DVC-tracked
├── dvc.yaml                   # DVC pipeline stages (dvc repro runs it end-to-end)
├── requirements.txt
├── micro_data_preparation_yaml.py
├── transactions_data_main_parquet_yaml.py
├── transactions_data_preparation_mode_parquet_yaml.py
├── regression_modeling_log_parquet_yaml_multi.py
├── forecasting_engine_chronos.py
├── forecasting_engine_chronos_news.py
├── data/
│   ├── raw/            <- DVC-tracked, pulled from s3://truestates-ml-ops/raw
│   └── processed/      <- DVC-tracked stage outputs
└── src/
    ├── utils/
    │   ├── mlflow_utils.py     # dagshub.init() + MLflow bootstrap, param logging
    │   └── drift_utils.py      # data drift (KS-test + PSI) & model drift checks
    ├── ingestion/run.py
    ├── cleaning/run.py
    ├── merging/run.py
    ├── modeling/run.py
    ├── forecasting/run.py
    └── forecasting_news/run.py
```

## 2. One-time setup

```bash
git clone https://dagshub.com/AnanthS/truestates-ml-ops.git
cd truestates-ml-ops
pip install -r requirements.txt

# --- DVC remote pointing at DagsHub's S3-compatible bucket storage ---
dvc init
dvc remote add origin s3://truestates-ml-ops -d
dvc remote modify origin endpointurl https://dagshub.com/AnanthS/truestates-ml-ops.s3
dvc remote modify origin --local access_key_id     <your_dagshub_token>
dvc remote modify origin --local secret_access_key <your_dagshub_token>

# --- pull existing raw files that are already in the bucket (projects.parquet, etc.) ---
dvc pull -r origin data/raw

# --- MLflow credentials (DagsHub personal access token) ---
export DAGSHUB_REPO_OWNER=AnanthS
export DAGSHUB_REPO_NAME=truestates-ml-ops
export DAGSHUB_TOKEN=<your_dagshub_token>       # used by dagshub.init() for auth
```

`dagshub.init(..., mlflow=True)` in `src/utils/mlflow_utils.py` automatically sets
`MLFLOW_TRACKING_URI` to `https://dagshub.com/AnanthS/truestates-ml-ops.mlflow`
and authenticates using `DAGSHUB_TOKEN` — no manual `mlflow.set_tracking_uri()` needed.

## 3. Running the pipeline

Full run (all 6 stages, matches the flowchart order):
```bash
python main.py
```

Run individual stages:
```bash
python main.py Ingestion Cleaning Merging
```

Run as a reproducible DVC pipeline (recommended — this also stages DVC outputs
automatically after each step and skips stages whose deps haven't changed):
```bash
dvc repro
dvc push -r origin       # push new/changed data + model artifacts to the bucket
git add dvc.lock && git commit -m "pipeline run" && git push
```

## 4. What gets tracked where

| Stage | DVC (data/model versioning) | MLflow (experiment tracking) |
|---|---|---|
| Ingestion | — | ingestion metadata: row/col counts, config params |
| Cleaning | — | cleaning params: filters, row counts before/after |
| Merging | merged dataset (`latest_combined_data.parquet`) | merge stats, null %, outliers file, **data drift report** |
| Modeling | train/test split outputs + model artifacts (`models/`) | params, metrics (`all_area_metrics.csv`), model files, **model drift check** |
| Forecasting | forecast CSV (`final_chronos_forecasts.csv`) | prediction artifact, backtest metrics |
| Forecasting_news | alternate forecast CSV (`adjusted_macro_forecast.csv`) | alternate run tracking, news context artifact |

## 5. Drift monitoring details

- **Data drift** (`src/utils/drift_utils.py::check_data_drift`): runs at the
  Merging stage. Compares numeric columns (configured in `config.yaml -> drift.numeric_cols_for_data_drift`)
  between the previous merged snapshot (`latest_combined_data.parquet.prev`, itself
  DVC/MLflow tracked) and the current run, using a KS-test (p < 0.05) and PSI (> 0.2)
  as drift flags. Results are logged as MLflow metrics + a JSON artifact.
- **Model drift** (`src/utils/drift_utils.py::check_model_drift`): runs at the
  Modeling stage. Pulls the most recent prior run in the same MLflow experiment
  via `MlflowClient.search_runs`, compares the configured metric
  (`config.yaml -> drift.model_metric_key`, default `r2`), and flags drift if
  performance degrades more than 10%.

## 6. Notes / things to double check before first run

- `config.yaml` above only shows the **paths + mlflow + drift** sections updated
  for DagsHub/DVC. Paste back in your original `ingestion_columns`,
  `transaction_settings`, `merging_params`, `model_params`, `training_columns`,
  `training_logic`, `market_mappings`, `forecast_settings`, `forecasting_settings`,
  `archive`, `macro_news_settings`, and `area_mapping` blocks unchanged — nothing
  in those needed to change for the migration.
- Your original modules (`run_ingestion`, `run_transaction_processing`, etc.)
  are called exactly as before; only orchestration + tracking is new, so no
  internal pipeline logic needed to change.
- Excel scorecards + raw parquet files should live under `data/raw/` and be
  DVC-tracked (`dvc add data/raw/*.parquet data/raw/*.xlsx`) so they map onto
  the `raw/` bucket folder shown in your DagsHub Files tab.
- Never commit `DAGSHUB_TOKEN` — use env vars or DagsHub's `dagshub.auth` login flow.
