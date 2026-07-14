"""
Lightweight data-drift and model-drift utilities.
Data drift: KS-test / PSI between the previous DVC-tracked merged dataset
            and the newly produced one, logged to MLflow.
Model drift: compares current run metrics to the best/most recent
            registered run's metrics via the MLflow client, logged as deltas.
"""
import json
import logging
import numpy as np
import pandas as pd
import mlflow
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD_PVALUE = 0.05      # KS-test: p < 0.05 -> drift flagged
PSI_THRESHOLD = 0.2                # PSI > 0.2 -> significant drift
METRIC_DRIFT_THRESHOLD_PCT = 0.10  # 10% degradation vs previous best run


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e_pct, _ = np.histogram(expected, bins=breakpoints)
    a_pct, _ = np.histogram(actual, bins=breakpoints)
    e_pct = np.where(e_pct == 0, 1e-6, e_pct) / len(expected)
    a_pct = np.where(a_pct == 0, 1e-6, a_pct) / len(actual)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def check_data_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                      numeric_cols: list, artifact_name: str = "data_drift_report.json"):
    """Runs KS-test + PSI per numeric column and logs a report + metrics to MLflow."""
    report = {}
    n_drifted = 0
    for col in numeric_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        base = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
        curr = pd.to_numeric(current_df[col], errors="coerce").dropna().values
        if len(base) < 30 or len(curr) < 30:
            continue
        stat, pvalue = ks_2samp(base, curr)
        psi = _psi(base, curr)
        drifted = bool(pvalue < DRIFT_THRESHOLD_PVALUE or psi > PSI_THRESHOLD)
        n_drifted += int(drifted)
        report[col] = {"ks_stat": float(stat), "p_value": float(pvalue),
                        "psi": psi, "drift_flag": drifted}
        mlflow.log_metric(f"drift_ks_pvalue_{col}", float(pvalue))
        mlflow.log_metric(f"drift_psi_{col}", float(psi) if not np.isnan(psi) else -1.0)

    overall_drift = n_drifted > 0
    mlflow.log_metric("data_drift_columns_flagged", n_drifted)
    mlflow.log_metric("data_drift_detected", int(overall_drift))

    with open(artifact_name, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(artifact_name)
    logger.info(f"[DATA DRIFT] {n_drifted}/{len(numeric_cols)} columns flagged. "
                f"Overall drift={overall_drift}")
    return overall_drift, report


def check_model_drift(current_metrics: dict, experiment_name: str,
                       metric_key: str = "r2", client=None):
    """Compares current run's metric against the previous best run in the same
    MLflow experiment and logs the delta + a drift flag."""
    client = client or mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning("No previous experiment found; skipping model drift check.")
        return False, None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=5,
    )
    prev_metric = None
    for r in runs:
        if metric_key in r.data.metrics:
            prev_metric = r.data.metrics[metric_key]
            break

    if prev_metric is None:
        logger.info("No historical run with comparable metric; baseline established.")
        mlflow.log_metric("model_drift_detected", 0)
        return False, None

    current_val = current_metrics.get(metric_key)
    if current_val is None:
        return False, None

    pct_change = (current_val - prev_metric) / (abs(prev_metric) + 1e-9)
    drifted = bool(pct_change < -METRIC_DRIFT_THRESHOLD_PCT)

    mlflow.log_metric(f"model_drift_prev_{metric_key}", prev_metric)
    mlflow.log_metric(f"model_drift_pct_change_{metric_key}", pct_change)
    mlflow.log_metric("model_drift_detected", int(drifted))
    logger.info(f"[MODEL DRIFT] {metric_key}: prev={prev_metric:.4f} "
                f"current={current_val:.4f} change={pct_change:.2%} drift={drifted}")
    return drifted, pct_change
