# 🏗️ TruEstates ML-Ops: Technical Pipeline Documentation

## 1. Executive Summary

The **TruEstates ML-Ops Pipeline** is an end-to-end, cloud-native machine learning architecture designed to ingest, clean, merge, and model Dubai real estate transaction data. It forecasts future price trajectories using state-of-the-art time-series models (Chronos) and dynamically adjusts predictions based on live macroeconomic news sentiment.

The system utilizes a stream-based storage architecture via **DagsHub's S3-compatible storage**, entirely eliminating the need for local disk caching and manual folder management during execution.

---

## 2. System Architecture & Tracking

* **Storage Layer:** DagsHub S3-compatible Bucket (`s3://truestates-ml-ops/data`).
* **Data Access Pattern:** Stream-based I/O using `dagshub.get_repo_bucket_client("...", flavor="s3fs")`.
* **Experiment Tracking (MLflow):** Automatically initialized via `dagshub.init()`. Logs parameters, metrics, model artifacts, and data/model drift.
* **Data Versioning (DVC):** Tracks dataset states and artifact hashes across runs.
* **Orchestration:** `main.py` triggers individual modular stages sequentially.

---

## 3. Repository Layout

The repository is structured to separate orchestration, configuration, pipeline execution, and modular utilities.

```text
truestates-ml-ops/
├── main.py                                           # Global orchestrator
├── config.yaml                                       # Unified S3 paths & hyperparameters
├── dvc.yaml                                          # DVC pipeline stage definitions
├── requirements.txt                                  # Environment dependencies
│
├── micro_data_preparation_yaml.py                    # Stage 1: Ingestion
├── transactions_data_main_parquet_yaml.py            # Stage 2: Cleaning
├── transactions_data_preparation_mode_parquet_yaml.py# Stage 3: Merging
├── regression_modeling_log_parquet_yaml_multi.py     # Stage 4: Modeling
├── forecasting_engine_chronos.py                     # Stage 5: Forecasting
├── forecasting_engine_chronos_news.py                # Stage 6: Macro-News
│
├── data/                                             
│   ├── raw/                                          # DVC-tracked raw inputs (projects.parquet, etc.)
│   └── processed/                                    # DVC-tracked stage outputs
│
└── src/
    ├── utils/
    │   ├── mlflow_utils.py                           # DagsHub/MLflow bootstrap & auth
    │   └── drift_utils.py                            # Data drift (KS-test/PSI) & Model drift checks
    ├── ingestion/run.py                              # Sub-module router
    ├── cleaning/run.py                               # Sub-module router
    ├── merging/run.py                                # Sub-module router
    ├── modeling/run.py                               # Sub-module router
    ├── forecasting/run.py                            # Sub-module router
    └── forecasting_news/run.py                       # Sub-module router

```

---

## 4. Pipeline Execution Flow

The pipeline executes sequentially through 6 core modules. Because of the `s3fs` refactor, each step streams data natively from the DagsHub S3 bucket into memory, processes it, and streams the output directly back.

| Step | Stage Name | Primary Objective | Output Artifacts | Tracking (MLflow / DVC) |
| --- | --- | --- | --- | --- |
| **1** | **Ingestion** | Aggregates raw Micro Data (Projects, Developers, Buildings, Units) into a unified framework. | `latest_pdbu_df.csv` | MLflow: Metadata |
| **2** | **Cleaning** | Filters invalid transactions, standardizes dates, and formats rooms. | `unit_res_trans_16.parquet` | MLflow: Cleaning Params |
| **3** | **Merging** | Fuses Micro Data and Transactions. Integrates Excel Scorecards and handles outliers. | `latest_combined_data.parquet` | DVC: Data Hash<br>

<br>MLflow: Data Drift (KS-Test/PSI) |
| **4** | **Modeling** | Trains 27 individual area-specific regression models to predict square-meter sale prices. | `.joblib` models<br>

<br>`all_area_metrics.csv` | DVC: Artifacts<br>

<br>MLflow: Model Drift (R²) |
| **5** | **Forecasting** | Generates 6-month baseline price predictions using Chronos time-series pipelines. | `final_chronos_forecasts.csv` | DVC: Output CSV<br>

<br>MLflow: Prediction Artifact |
| **6** | **Macro-News** | Scrapes live RSS feeds, synthesizes macro events via LLMs, and applies dynamic decay modifiers. | `adjusted_macro_forecast.csv` | DVC: Output CSV<br>

<br>MLflow: Alternate Run Tracking |

---

## 5. Google Colab Execution Guide

Because the pipeline relies on DagsHub for both Git tracking and S3 streaming, the execution flow in ephemeral environments like Google Colab must be set up securely.

### Step 1: Securely Clone the Repository

Avoid using Colab's virtual mount for execution, as it can cause `Input/output error` and lock files. Instead, clone directly to the Colab hard drive.

```python
# In a Colab cell
import os

# Set up secure DagsHub clone URL using your Personal Access Token
TOKEN = "your_dagshub_personal_access_token"
REPO_URL = f"https://poojariprakash88:{TOKEN}@dagshub.com/poojariprakash88/truestates-ml-ops.git"

# Clone the latest code and move inside the folder
!git clone {REPO_URL}
%cd truestates-ml-ops

```

### Step 2: Install Dependencies

Install the required packages, including `s3fs` and `catboost`.

```python
!pip install -r requirements.txt

```

### Step 3: Configure Authentication Environment Variables

The pipeline utilizes `dagshub.init()` and `s3fs`. You must export your credentials so Pandas and MLflow can seamlessly authenticate with the S3 bucket in the background.

```python
import os

TOKEN = "your_dagshub_personal_access_token"

# 1. DagsHub API & MLflow Auth
os.environ["DAGSHUB_TOKEN"] = TOKEN
os.environ["DAGSHUB_REPO_OWNER"] = "poojariprakash88"
os.environ["DAGSHUB_REPO_NAME"] = "truestates-ml-ops"

# 2. S3FS Auth for Pandas
os.environ["AWS_ACCESS_KEY_ID"] = TOKEN
os.environ["AWS_SECRET_ACCESS_KEY"] = TOKEN
os.environ["AWS_ENDPOINT_URL_S3"] = "https://dagshub.com/poojariprakash88/truestates-ml-ops.s3"

```

### Step 4: Run the Pipeline

With the environment configured, the python scripts will stream the data natively. No `dvc pull` is required for the data processing stages to succeed.

```bash
# Run the full end-to-end pipeline
!python main.py

# OR run specific stages
!python main.py Modeling Forecasting Forecasting_news

```

---

## 6. Developer Standards (Standard Operating Procedures)

To maintain pipeline stability and prevent `FileNotFoundError` crashes on cloud runners, all contributors **must** adhere to the following file I/O standards:

### 🚫 Anti-Pattern: Local Disk Reads

Standard Pandas calls assume a local hard drive. *Do not use this approach:*

```python
# BAD: Will crash because S3 does not exist on the local disk
df = pd.read_parquet(config['paths']['merging_output']) 

```

### ✅ Standard: S3 Stream Reads

Always initialize the DagsHub client and wrap reads in an open stream (`rb` for reading binary parquet/joblib, `r` for CSV/JSON).

```python
# GOOD: Streams the data securely into memory
with fs.open(config['paths']['merging_output'], "rb") as f:
    df = pd.read_parquet(f)

```

### ✅ Standard: S3 Stream Writes

S3 does not require `os.makedirs`. Simply open the destination path in write mode (`wb` for parquet/joblib, `w` for CSV/JSON).

```python
# GOOD: Pushes the file straight to the bucket
with fs.open(config['paths']['models_output'], "wb") as f:
    joblib.dump(model, f)

```
