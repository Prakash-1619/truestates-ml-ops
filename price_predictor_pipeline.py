import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CODE_DIR / "config.yaml"


def load_config():
    import yaml
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()
BASE_PATH = Path(config["paths"]["base_dir"]).resolve()
MODEL_DIR = BASE_PATH / config["paths"].get("models_dir", "models")
COL_DIR = BASE_PATH / config["paths"].get("columns_dir", "trained_columns")
ARCHIVE_DIR = BASE_PATH / config["paths"].get("old_files_dir", "old_files")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
COL_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_MODEL_AREAS = set(config.get("market_mappings", {}).get("direct_areas", []))
PROXY_MAPPING = {}
for group_name, areas in config.get("market_mappings", {}).get("groups", {}).items():
    for a in areas:
        PROXY_MAPPING[a] = group_name
for proxy_name, areas in config.get("market_mappings", {}).get("proxies", {}).items():
    for a in areas:
        PROXY_MAPPING[a] = proxy_name

CATEGORICAL_FEATURES = config.get("training_columns", {}).get("cat_cols", [])
NUMERIC_FEATURES = config.get("training_columns", {}).get("num_cols", [])


def get_slug(name):
    return str(name).replace(" ", "_").replace("'", "")


def safe_load(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None


def load_assets(model_key):
    slug = get_slug(model_key)
    candidates = [slug, slug.lower(), slug.upper(), slug.title()]
    model = None
    cols = None

    for s in candidates:
        model_path = MODEL_DIR / f"best_model_{s}.joblib"
        col_path = COL_DIR / f"trained_columns_{s}.joblib"
        if model is None and model_path.exists():
            model = safe_load(model_path)
        if cols is None and col_path.exists():
            cols = safe_load(col_path)
        if model is not None and cols is not None:
            break

    if model is None or cols is None:
        old_model_candidates = [ARCHIVE_DIR / f"old_best_model_{s}.joblib" for s in candidates]
        old_col_candidates = [ARCHIVE_DIR / f"old_trained_columns_{s}.joblib" for s in candidates]
        for mp in old_model_candidates:
            if model is None and mp.exists():
                model = safe_load(mp)
        for cp in old_col_candidates:
            if cols is None and cp.exists():
                cols = safe_load(cp)

    if model is None:
        raise FileNotFoundError(f"Missing model for {model_key}")
    if cols is None:
        raise FileNotFoundError(f"Missing trained columns for {model_key}")
    return model, cols


def _build_input_frame(input_data, train_columns):
    raw = pd.DataFrame([input_data])
    raw = raw.reindex(columns=["area_name"] + CATEGORICAL_FEATURES + NUMERIC_FEATURES, fill_value=np.nan)

    for c in CATEGORICAL_FEATURES:
        if c in raw.columns:
            raw[c] = raw[c].fillna("Unknown").astype(str)

    for c in NUMERIC_FEATURES:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    encoded = pd.get_dummies(
        raw.drop(columns=["area_name"], errors="ignore"),
        columns=[c for c in CATEGORICAL_FEATURES if c in raw.columns],
        drop_first=False,
    ).astype(float)
    encoded = encoded.reindex(columns=train_columns, fill_value=0)
    return raw, encoded


def _prepare_series(df, model_key):
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    if "area" in df.columns:
        df = df[df["area"].astype(str) == model_key].copy()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.sort_values("month")
    return df


def predict_property_price(input_data, forecast_df=None, historic_df=None):
    area = input_data["area_name"]

    if area in DIRECT_MODEL_AREAS:
        model_key = area
    elif area in PROXY_MAPPING:
        model_key = PROXY_MAPPING[area]
    else:
        raise ValueError(f"No model/proxy mapping found for: {area}")

    model, train_columns = load_assets(model_key)
    _, encoded_input = _build_input_frame(input_data, train_columns)
    raw_input = pd.DataFrame([input_data]).drop(columns=["area_name"], errors="ignore")

    use_encoded = hasattr(model, "n_features_in_") and getattr(model, "n_features_in_", None) == len(train_columns)
    prediction_input = encoded_input if use_encoded else raw_input.reindex(
        columns=CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        fill_value=np.nan,
    )

    for c in CATEGORICAL_FEATURES:
        if c in prediction_input.columns:
            prediction_input[c] = prediction_input[c].fillna("Unknown").astype(str)
    for c in NUMERIC_FEATURES:
        if c in prediction_input.columns:
            prediction_input[c] = pd.to_numeric(prediction_input[c], errors="coerce").fillna(0)

    base_prediction = float(model.predict(prediction_input)[0])

    if forecast_df is None:
        forecast_df = pd.DataFrame()
    if historic_df is None:
        historic_df = pd.DataFrame()

    if not forecast_df.empty and "month" in forecast_df.columns:
        forecast_df = forecast_df.copy()
        forecast_df["month"] = pd.to_datetime(forecast_df["month"], errors="coerce")
    if not historic_df.empty and "month" in historic_df.columns:
        historic_df = historic_df.copy()
        historic_df["month"] = pd.to_datetime(historic_df["month"], errors="coerce")

    if forecast_df.empty and historic_df.empty:
        return pd.DataFrame({"month": [pd.Timestamp.now()], "median_price": [base_prediction], "area": [area]})

    forecast_df = _prepare_series(forecast_df, model_key)
    historic_df = _prepare_series(historic_df, model_key)

    if not forecast_df.empty:
        if "growth_factor" in forecast_df.columns:
            forecast_df["median_price"] = base_prediction * forecast_df["growth_factor"]
        elif "predictions" in forecast_df.columns:
            forecast_df["median_price"] = forecast_df["predictions"]
        else:
            forecast_df["median_price"] = base_prediction

    if not historic_df.empty:
        historic_df["median_price"] = base_prediction

    combined = pd.concat([historic_df, forecast_df], ignore_index=True)
    combined["area"] = area
    if "median_price" not in combined.columns:
        combined["median_price"] = base_prediction
    if "month" not in combined.columns:
        combined["month"] = pd.Timestamp.now()

    return combined[["month", "median_price", "area"]].sort_values("month")


if __name__ == "__main__":
    logger.info("price_predictor_pipeline.py loaded")