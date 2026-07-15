import pandas as pd
import numpy as np
import joblib
import os
import re
import logging
import yaml
import time
import shutil
import xgboost as xgb
import catboost as cb
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from dagshub import get_repo_bucket_client
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pd.set_option('io.parquet.engine', 'pyarrow')

def ensure_old_dir(base_dir, config):
    folder = config.get('archive', {}).get('folder_name', 'old_files')
    return f"{base_dir}/{folder}"

def archive_existing_file(file_path, old_dir, prefix="old_"):
    if fs.exists(file_path):
        filename = file_path.split('/')[-1]
        archived_path = f"{old_dir}/{prefix}{filename}"
        fs.rename(file_path, archived_path)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    base = config['paths']['base_dir'].replace("s3://", "")
    
    # Natively construct the paths using S3 format instead of os.path.join
    config['paths']['input_full'] = f"{base}/{config['paths']['model_input']}"
    config['paths']['m_dir'] = f"{base}/{config['paths']['models_dir']}"
    config['paths']['c_dir'] = f"{base}/{config['paths']['columns_dir']}"
    config['paths']['metrics_path'] = f"{base}/{config['paths'].get('metrics_file', 'processed/all_area_metrics.csv')}"
    config['paths']['ranges_path'] = f"{base}/{config['paths'].get('ranges_file', 'model_requirements/input_ranges.csv')}"
    config['paths']['old_dir'] = f"{base}/{config['paths'].get('old_files_dir', 'old_files')}"
    return config

def train_and_save(name, area_df, target, config, cat_cols, num_cols, date_col='instance_date'):
    if date_col in area_df.columns:
        max_date = area_df[date_col].max()
        test_df = area_df[(area_df[date_col].dt.year == max_date.year) & (area_df[date_col].dt.month == max_date.month)]
        if not test_df.empty:
            train_df = area_df[area_df[date_col] < test_df[date_col].min()]
        else:
            train_df = pd.DataFrame()

    if test_df.empty or train_df.empty:
        logger.warning(f"⚠️ Not enough data for Train/Test split for {name} (Requires historical data). Skipping...")
        return {}, {}, []

    combined_df = pd.concat([train_df, test_df])

    existing_cat = [c for c in cat_cols if c in combined_df.columns]
    existing_num = [c for c in num_cols if c in combined_df.columns]

    X_raw = combined_df[existing_cat + existing_num].copy()
    X_raw[existing_cat] = X_raw[existing_cat].fillna('Unknown').astype(str)
    X_raw[existing_num] = X_raw[existing_num].fillna(0).astype(float)

    cat_indices = [X_raw.columns.get_loc(c) for c in existing_cat]
    X_encoded = pd.get_dummies(X_raw, columns=existing_cat).astype(float)

    train_len = len(train_df)

    X_train_raw = X_raw.iloc[:train_len]
    X_test_raw = X_raw.iloc[train_len:]

    X_train_enc = X_encoded.iloc[:train_len]
    X_test_enc = X_encoded.iloc[train_len:]

    y_train = train_df[target].values
    y_test = test_df[target].values

    summary_dict = {'area_name_en': name, 'pre_encode_columns': str(list(X_raw.columns))}
    for col in existing_cat + existing_num:
        summary_dict[col] = str(sorted(combined_df[col].dropna().unique().tolist())[:10])
    if 'procedure_area' in combined_df.columns:
        summary_dict['procedure_area_min'] = combined_df['procedure_area'].min()
        summary_dict['procedure_area_max'] = combined_df['procedure_area'].max()

    models_and_params = {
        'CatBoost': {
            'model': cb.CatBoostRegressor(random_state=42, verbose=0, thread_count=-1),
            'params': {
                'regressor__iterations': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__depth': [4, 6, 8]
            },
            'use_encoded': False
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 6, 9]
            },
            'use_encoded': True
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            },
            'use_encoded': True
        }
    }

    groups_requiring_log = config.get('forecast_settings', {}).get('log_transform_groups', [])
    apply_log_transform = name in groups_requiring_log
    target_func = np.log1p if apply_log_transform else None
    target_inv = np.expm1 if apply_log_transform else None

    scoring_metrics = {
        'mape': 'neg_mean_absolute_percentage_error',
        'r2': 'r2'
    }

    best_area_mape = float('inf')
    best_area_model = None
    best_area_metrics = {}
    param_logs = []

    for model_name, mp in models_and_params.items():
        X_tr_use = X_train_enc if mp['use_encoded'] else X_train_raw
        X_te_use = X_test_enc if mp['use_encoded'] else X_test_raw

        ttr = TransformedTargetRegressor(regressor=mp['model'], func=target_func, inverse_func=target_inv)

        search = RandomizedSearchCV(
            estimator=ttr,
            param_distributions=mp['params'],
            n_iter=5,
            cv=3,
            scoring=scoring_metrics,
            refit='mape',
            random_state=42,
            n_jobs=-1 if model_name != 'CatBoost' else 1
        )

        fit_params = {}
        if model_name == 'CatBoost':
            fit_params['cat_features'] = cat_indices

        search.fit(X_tr_use, y_train, **fit_params)
        cv_res = search.cv_results_

        for i in range(len(cv_res['params'])):
            param_logs.append({
                'area': name,
                'algorithm': model_name,
                'params': str(cv_res['params'][i]),
                'cv_mean_mape': -cv_res['mean_test_mape'][i],
                'cv_mean_r2': cv_res['mean_test_r2'][i],
                'cv_rank': cv_res['rank_test_mape'][i]
            })

        current_best = search.best_estimator_
        test_preds = current_best.predict(X_te_use)

        if len(y_test) > 0:
            test_mape = mean_absolute_percentage_error(y_test, test_preds)

            if test_mape < best_area_mape:
                best_area_mape = test_mape
                best_area_model = current_best

                train_preds = current_best.predict(X_tr_use)

                best_area_metrics = {
                    'area': name,
                    'best_algorithm': model_name,
                    'best_params': str(search.best_params_),
                    'test_r2': r2_score(y_test, test_preds) if len(y_test) > 1 else np.nan,
                    'test_mape': test_mape,
                    'test_mae': mean_absolute_error(y_test, test_preds),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
                    'train_r2': r2_score(y_train, train_preds),
                    'train_mape': mean_absolute_percentage_error(y_train, train_preds),
                    'train_mae': mean_absolute_error(y_train, train_preds),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
                    'test_samples': len(y_test),
                    'train_samples': len(y_train),
                    'train_shape': X_tr_use.shape,
                    'median_actual_price': np.median(y_test) if len(y_test) > 0 else np.nan,
                    'median_pred_price': np.median(test_preds) if len(test_preds) > 0 else np.nan
                }

    if best_area_model is not None:
        clean_n = re.sub(r'\W+', '_', str(name))
        model_filepath = os.path.join(config['paths']['m_dir'], f"best_model_{clean_n}.joblib")
        cols_filepath = os.path.join(config['paths']['c_dir'], f"trained_columns_{clean_n}.joblib")

        old_dir = config['paths'].get('old_dir') or ensure_old_dir(config['paths']['base_dir'], config)
        prefix = config.get('archive', {}).get('prefix', 'old_')
        archive_existing_file(model_filepath, old_dir, prefix)
        archive_existing_file(cols_filepath, old_dir, prefix)

        final_cols = list(X_train_raw.columns) if best_area_metrics['best_algorithm'] == 'CatBoost' else list(X_train_enc.columns)

        with fs.open(model_filepath, "wb") as f:
            joblib.dump(best_area_model, f)
        with fs.open(cols_filepath, "wb") as f:
            joblib.dump(final_cols, f)

    return best_area_metrics, summary_dict, param_logs

def run_model_training():
    config = load_config()
    logger.info("🚀 Starting ML Training Pipeline...")

    cat_cols = config.get('training_columns', {}).get('cat_cols', [])
    num_cols = config.get('training_columns', {}).get('num_cols', [])

    if not cat_cols and not num_cols:
        logger.error("❌ 'training_columns' not found in config.yaml! Please add them.")
        return

    logger.info(f"🔍 Base Features Loaded: {len(cat_cols)} Categorical, {len(num_cols)} Numerical")
    logger.info("=======================================================================================")

    with fs.open(config['paths']['input_full'], "rb") as f:
        df_raw = pd.read_parquet(f)

    trans_filters = config.get('training_logic', {}).get('trans_group_filter', None)
    if trans_filters and 'trans_group_en' in df_raw.columns:
        before_len = len(df_raw)
        df_raw = df_raw[df_raw['trans_group_en'].isin(trans_filters)]
        logger.info(f"🔄 Filtered trans_group_en for {trans_filters}. Kept {len(df_raw)} / {before_len} records.")

    target_col = config.get('training_logic', {}).get('target_col', 'meter_sale_price')
    if target_col not in df_raw.columns:
        logger.warning(f"⚠️ Target '{target_col}' not found. Defaulting to 'actual_price'.")
        target_col = 'actual_price' if 'actual_price' in df_raw.columns else 'transaction_value'
    df_raw = df_raw.dropna(subset=[target_col])

    date_col = 'instance_date'
    if date_col in df_raw.columns:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce')
        df_raw = df_raw.dropna(subset=[date_col])

        start_date_str = config.get('training_logic', {}).get('start_date', None)
        if start_date_str:
            try:
                start_date = pd.to_datetime(start_date_str)
                before_len = len(df_raw)
                df_raw = df_raw[df_raw[date_col] >= start_date]
                logger.info(f"📅 Filtered records strictly from {start_date_str} onwards. Kept {len(df_raw)} / {before_len} records.")
            except Exception as e:
                logger.error(f"❌ Invalid 'start_date' format in config: {start_date_str}. Error: {e}")

    market_mappings = config.get('market_mappings', {})
    mapping_groups = market_mappings.get('groups', {})
    mapping_proxies = market_mappings.get('proxies', {})
    combined_mappings = {**mapping_groups, **mapping_proxies}

    min_samples = config.get('training_logic', {}).get('min_samples_to_train', 30)
    area_counts = df_raw['area_name_en'].value_counts()
    individual_areas_to_run = area_counts[area_counts >= 6000].index.tolist()

    best_results, summaries, all_param_logs = [], [], []

    logger.info(f"--- 🚀 Starting Individual Area Models ({len(individual_areas_to_run)} total) ---")
    for area in individual_areas_to_run:
        area_df = df_raw[df_raw['area_name_en'] == area]

        if len(area_df) > min_samples:
            metrics, summary, params = train_and_save(area, area_df.copy(), target_col, config, cat_cols, num_cols, date_col)
            if metrics:
                best_results.append(metrics)
                summaries.append(summary)
                all_param_logs.extend(params)

                disp_name = (area[:18] + '..') if len(area) > 20 else area
                shape_str = str(metrics.get('train_shape', 'N/A'))
                logger.info(f"🏆 {disp_name:20} | Shape: {shape_str:10} | Algo: {metrics.get('best_algorithm', 'N/A'):10} | R2: {metrics.get('test_r2', 0):5.2f} | MAPE: {metrics.get('test_mape', 0):6.2%} | MAE: {metrics.get('test_mae', 0):,.0f} | RMSE: {metrics.get('test_rmse', 0):,.0f}")
        else:
            logger.warning(f"⚠️ Skipped {area}: Insufficient train data ({len(area_df)} rows)")

    logger.info(f"--- 🚀 Starting Combined Models ({len(combined_mappings)} total) ---")
    for combined_name, combined_area_list in combined_mappings.items():
        area_df = df_raw[df_raw['area_name_en'].isin(combined_area_list)]

        if len(area_df) > min_samples:
            metrics, summary, params = train_and_save(combined_name, area_df.copy(), target_col, config, cat_cols, num_cols, date_col)
            if metrics:
                best_results.append(metrics)
                summaries.append(summary)
                all_param_logs.extend(params)

                shape_str = str(metrics.get('train_shape', 'N/A'))
                logger.info(f"🏆 {combined_name:20} | Shape: {shape_str:10} | Algo: {metrics.get('best_algorithm', 'N/A'):10} | R2: {metrics.get('test_r2', 0):5.2f} | MAPE: {metrics.get('test_mape', 0):6.2%} | MAE: {metrics.get('test_mae', 0):,.0f} | RMSE: {metrics.get('test_rmse', 0):,.0f}")
        else:
            logger.warning(f"⚠️ Skipped {combined_name}: Insufficient train data ({len(area_df)} rows)")

    base = config['paths']['base_dir']
    metrics_path = config['paths'].get('metrics_path') or os.path.join(base, config['paths'].get('metrics_file', 'best_model_metrics.csv'))
    ranges_path = config['paths'].get('ranges_path') or os.path.join(base, config['paths'].get('ranges_file', 'model_ranges.csv'))
    params_path = os.path.join(base, 'parameter_tuning_log.csv')

    old_dir = config['paths'].get('old_dir') or ensure_old_dir(base, config)
    prefix = config.get('archive', {}).get('prefix', 'old_')

    archive_existing_file(metrics_path, old_dir, prefix)
    archive_existing_file(ranges_path, old_dir, prefix)
    archive_existing_file(params_path, old_dir, prefix)

    if best_results:
        with fs.open(metrics_path, "w") as f:
            pd.DataFrame(best_results).to_csv(f, index=False)
        with fs.open(ranges_path, "w") as f:
            pd.DataFrame(summaries).to_csv(f, index=False)
        with fs.open(params_path, "w") as f:
            pd.DataFrame(all_param_logs).to_csv(f, index=False)
        logger.info("🎉 Training Complete. All metrics and tuning logs saved to base directory.")
    else:
        logger.error("❌ No models were successfully trained.")

if __name__ == "__main__":
    run_model_training()
