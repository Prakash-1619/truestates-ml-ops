Here's the complete guide updated with your actual repo: `https://dagshub.com/poojariprakash88/truestates-ml-ops`

## 1. Clone the repo and unzip the pipeline

Since DagsHub requires token authentication for cloning (as you saw earlier), embed your token in the clone URL:

```bash
git clone https://poojariprakash88:<your_dagshub_token>@dagshub.com/poojariprakash88/truestates-ml-ops.git
cd truestates-ml-ops
# unzip truestates-ml-ops-pipeline.zip contents here, alongside your existing module files
unzip /path/to/truestates-ml-ops-pipeline.zip -d .
```

Get `<your_dagshub_token>` from `https://dagshub.com/user/settings/tokens` first.

## 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # on Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Get your DagsHub token

Go to your DagsHub profile → Settings → Tokens, and generate a personal access token. You'll use this for both DVC and MLflow authentication. [dagshub](https://dagshub.com/docs/integration_guide/dvc/)

## 4. Configure DVC remote (points at your raw bucket)

```bash
dvc init
dvc remote add origin s3://truestates-ml-ops -d
dvc remote modify origin endpointurl https://dagshub.com/poojariprakash88/truestates-ml-ops.s3
dvc remote modify origin --local access_key_id <your_dagshub_token>
dvc remote modify origin --local secret_access_key <your_dagshub_token>
```

The `--local` flag keeps your token out of git-tracked `.dvc/config`. [dagshub](https://dagshub.com/docs/integration_guide/dvc/)

Pull the raw files already sitting in your bucket (projects.parquet, developers.parquet, buildings.parquet, units.parquet, transactions.parquet, the three Excel scorecards):

```bash
dvc pull -r origin data/raw
```

## 5. Configure MLflow authentication

```bash
export DAGSHUB_REPO_OWNER=poojariprakash88
export DAGSHUB_REPO_NAME=truestates-ml-ops
export DAGSHUB_TOKEN=<your_dagshub_token>
```

On Windows PowerShell, use:

```powershell
$env:DAGSHUB_REPO_OWNER="poojariprakash88"
$env:DAGSHUB_REPO_NAME="truestates-ml-ops"
$env:DAGSHUB_TOKEN="<your_dagshub_token>"
```

`src/utils/mlflow_utils.py` calls `dagshub.init(repo_owner=..., repo_name=..., mlflow=True)`, which automatically sets `MLFLOW_TRACKING_URI` and authenticates — no manual `mlflow.set_tracking_uri()` needed. The first time you run it without `DAGSHUB_TOKEN` set, it may open a browser login instead. [dagshub](https://dagshub.com/docs/client/reference/setup.html)

## 6. Run the pipeline

**Option A — plain Python (runs main.py directly):**
```bash
python main.py
```
Or run specific stages only:
```bash
python main.py Ingestion Cleaning Merging
```

**Option B — DVC-managed run (recommended):**
```bash
dvc repro
```
This reads `dvc.yaml`, runs each stage in dependency order, skips stages whose inputs haven't changed, and automatically stages the outputs (merged dataset, models, forecast CSVs) for versioning. [zenn](https://zenn.dev/marcy_lab/articles/70bd98bda74f0c)

## 7. Push results back to DagsHub

```bash
dvc push -r origin
git add dvc.lock data/raw.dvc
git commit -m "Pipeline run: updated data + models"
git push
```

## 8. View results

- **MLflow experiments**: go to your DagsHub repo → Experiments tab, or open `https://dagshub.com/poojariprakash88/truestates-ml-ops.mlflow` directly to see params, metrics, drift flags, and artifacts per run. [dagshub](https://dagshub.com/DAGsHub-Official/dagshub-docs/src/636d68aa5ede5e94f44963855716a33bb6bf59ad/docs/integration_guide/giskard.md)
- **DVC pipeline graph**: DagsHub repo → Data Pipeline tab visualizes the `dvc.yaml` DAG with tracked outputs per stage. [dagshub](https://dagshub.com/docs/integration_guide/dvc/)









####################################################################################################################################################################3

cd C:\Users\pooja\truestates-ml-ops

python -m dvc destroy -f
python -m dvc init

git clone https://dagshub.com/poojariprakash88/truestates-ml-ops.git
cd truestates-ml-ops
pip install -r requirements.txt

python -m dvc remote add origin s3://truestates-ml-ops -f
python -m dvc remote modify origin endpointurl https://dagshub.com/poojariprakash88/truestates-ml-ops.s3
python -m dvc remote modify origin --local access_key_id <your_dagshub_token>
python -m dvc remote modify origin --local secret_access_key <your_dagshub_token>

python -m dvc remote default origin
python -m dvc pull -r origin data/raw

set DAGSHUB_REPO_OWNER=poojariprakash88
set DAGSHUB_REPO_NAME=truestates-ml-ops
set DAGSHUB_TOKEN=<your_dagshub_token>
