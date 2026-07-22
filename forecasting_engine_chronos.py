import os
import re
import logging
import shutil
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from chronos import Chronos2Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_old_dir(base_dir, config):
    folder = config.get('archive', {}).get('folder_name', 'old_files')
    old_dir = os.path.join(base_dir, folder)
    os.makedirs(old_dir, exist_ok=True)
    return old_dir

def archive_existing_file(file_path, old_dir, prefix="old_"):
    if os.path.exists(file_path):
        archived_path = os.path.join(old_dir, f"{prefix}{os.path.basename(file_path)}")
        shutil.move(file_path, archived_path)

class DubaiPropertyForecaster:
    def __init__(self, model_name="amazon/chronos-2", prediction_length=6, backtest_periods=6, zero_threshold=0.90):
        self.prediction_length = prediction_length
        self.backtest_periods = backtest_periods
        self.zero_threshold = zero_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Chronos Pipeline: {model_name} on {self.device}")
        self.pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=self.device)

        self.ROOM_MAP = {'1 B/R': 1, '2 B/R': 2, '3 B/R': 3, 'More than 3B/R': 4, 'Studio': 11, 'PENTHOUSE': 621}
        self.binary_cols = ["has_parking", "swimming_pool", "balcony", "elevator", "metro"]
        self.categorical_cols = ["rooms_num", "floor_bin", "nearest_landmark_en", "nearest_mall_en"]
        self.macro_cols = ["gold_close", "oil_close", "dfm_close"]
        self.tokeep_base = [
            'model_area_id', 'month', 'monthly_price',
            'gold_return_3m', 'gold_vol_3m',
            'dfm_return', 'dfm_return_3m', 'dfm_vol_3m',
            'lag_1_return', 'lag_2_return', 'log_txn_count',
            'oil_return_3m', 'oil_vol_3m'
        ]

    def _parse_rooms(self, val):
        if pd.isna(val):
            return np.nan
        v = str(val).strip()
        for k, num in self.ROOM_MAP.items():
            if k in v:
                return num
        m = re.search(r"\d+", v)
        return float(m.group()) if m else np.nan

    def _fetch_and_interpolate_macro(self, ticker_symbol, name_prefix, start_date, end_date):
        logger.info(f"Fetching {ticker_symbol} from yfinance...")
        df = yf.Ticker(ticker_symbol).history(
            start=start_date - pd.Timedelta(days=10),
            end=end_date + pd.Timedelta(days=10)
        ).reset_index()

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.rename(columns={'Date': 'date'})
        elif 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            df = df.rename(columns={'Datetime': 'date'})

        df = df.rename(columns={'Close': f'{name_prefix}_close', 'Volume': f'{name_prefix}_volume'})
        df = df[['date', f'{name_prefix}_close', f'{name_prefix}_volume']].set_index('date').sort_index()
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        return df.reindex(full_range).interpolate(method='linear').ffill().bfill()

    def _aggregate_month(self, group):
        result = {}
        txn_count = len(group)
        result["monthly_price"] = group["meter_sale_price"].median()
        result["txn_count"] = txn_count
        result["log_txn_count"] = np.log1p(txn_count)

        for col in self.binary_cols:
            result[f"pct_{col}"] = group[col].mean() if col in group.columns else 0.0

        for col in self.categorical_cols:
            if col not in group.columns:
                continue
            counts = group[col].value_counts()
            total = txn_count
            K = counts.shape[0]
            for category, count in counts.items():
                pct = (count + 1) / (total + K)
                safe_name = str(category).replace(" ", "_").replace("/", "").lower()
                result[f"pct_{col}_{safe_name}"] = pct

        for col in self.macro_cols:
            if col in group.columns:
                result[col] = group[col].median()

        return pd.Series(result)

    def _add_proxy_groups(self, trans_mapped):
        extra_dfs = []
        proxy_groups = [
            ("proxy1", ['Al Barshaa South Third', 'Al Barsha South Fourth', 'Al Yelayiss 2']),
            ("proxy2", ['Bukadara', 'Madinat Dubai Almelaheyah']),
            ("proxy3", ["Jabal Ali", "Me'Aisem First"])
        ]
        group_groups = [
            ("grouped1", ["Wadi Al Safa", "Al Kifaf"]),
            ("grouped2", ["Warsan Fourth", "Jabal Ali"]),
            ("grouped3", ["Zaabeel Second", "Zaabeel First"])
        ]

        for name, areas in proxy_groups + group_groups:
            g = trans_mapped[trans_mapped["area_name_en"].isin(areas)].copy()
            if not g.empty:
                g["area_name_en"] = name
                extra_dfs.append(g)

        return pd.concat([trans_mapped] + extra_dfs, ignore_index=True) if extra_dfs else trans_mapped

    def _aggregate_safe(self, df_copy):
        monthly_raw = df_copy.groupby(['area_name_en', 'month']).apply(self._aggregate_month)

        if isinstance(monthly_raw, pd.Series):
            monthly_df = monthly_raw.unstack().reset_index()
        else:
            monthly_df = monthly_raw.reset_index()

        monthly_df.columns = [str(c) for c in monthly_df.columns]
        monthly_df.columns.name = None

        if 'monthly_price' not in monthly_df.columns:
            if '0' in monthly_df.columns:
                monthly_df = monthly_df.rename(columns={'0': 'monthly_price'})
            elif 'level_2' in monthly_df.columns and '0' in monthly_df.columns:
                monthly_df = monthly_df.pivot_table(
                    index=['area_name_en', 'month'],
                    columns='level_2',
                    values='0',
                    aggfunc='first'
                ).reset_index()
                monthly_df.columns.name = None
                monthly_df.columns = [str(c) for c in monthly_df.columns]
                if '0' in monthly_df.columns:
                    monthly_df = monthly_df.rename(columns={'0': 'monthly_price'})

        if 'monthly_price' not in monthly_df.columns:
            raise ValueError(f"monthly_price missing after aggregation. Available columns: {list(monthly_df.columns)}")

        return monthly_df

    def _assign_model_and_saved_ids(self, monthly_df, source_df):
        if 'area_id' in source_df.columns:
            source_id_map = source_df[['area_name_en', 'area_id']].drop_duplicates()
            source_id_map = source_id_map.groupby('area_name_en')['area_id'].first().to_dict()
        else:
            source_id_map = {}

        proxy_map = {'proxy1': 'p1', 'proxy2': 'p2', 'proxy3': 'p3'}
        group_map = {'grouped1': 'g1', 'grouped2': 'g2', 'grouped3': 'g3'}

        def saved_id(name):
            if name in proxy_map:
                return proxy_map[name]
            if name in group_map:
                return group_map[name]
            if name in source_id_map:
                return source_id_map[name]
            return name

        monthly_df['actual_area_id'] = monthly_df['area_name_en'].map(source_id_map)
        monthly_df['area_id'] = monthly_df['area_name_en'].apply(saved_id)

        model_areas = monthly_df['area_name_en'].drop_duplicates().tolist()
        model_id_map = {area: idx for idx, area in enumerate(model_areas)}
        monthly_df['model_area_id'] = monthly_df['area_name_en'].map(model_id_map)

        return monthly_df, source_id_map, model_id_map

    def process_and_engineer_data(self, trans_path):
        logger.info("Initializing baseline transaction frame data prep...")
        if trans_path.endswith('.parquet'):
            df_copy = pd.read_parquet(trans_path)
        else:
            df_copy = pd.read_csv(trans_path)

        df_copy["date"] = pd.to_datetime(df_copy["instance_date"])
        df_copy["month"] = df_copy["date"].dt.to_period("M").apply(lambda r: r.start_time)
        df_copy.sort_values(by='month', inplace=True)
        df_copy["rooms_num"] = df_copy["rooms_en"].apply(self._parse_rooms)

        min_date, max_date = df_copy["date"].min(), df_copy["date"].max()
        oil_daily = self._fetch_and_interpolate_macro('CL=F', 'oil', min_date, max_date)
        gold_daily = self._fetch_and_interpolate_macro('GC=F', 'gold', min_date, max_date)
        dfm_daily = self._fetch_and_interpolate_macro('DFMGI.AE', 'dfm', min_date, max_date)

        macro_df = gold_daily.merge(oil_daily, left_index=True, right_index=True, how='inner')
        macro_df = macro_df.merge(dfm_daily, left_index=True, right_index=True, how='inner')

        df_copy = df_copy.merge(macro_df, left_on='date', right_index=True, how='inner')
        df_copy = df_copy.ffill()

        df_copy = self._add_proxy_groups(df_copy)

        logger.info("Running monthly group aggregations with full features footprint...")
        monthly_df = self._aggregate_safe(df_copy)
        monthly_df = monthly_df.sort_values(["area_name_en", "month"])

        monthly_df, source_id_map, model_id_map = self._assign_model_and_saved_ids(monthly_df, df_copy)

        monthly_df["log_price"] = np.log(monthly_df["monthly_price"].clip(lower=1e-9))
        monthly_df["future_log_return_1m"] = monthly_df.groupby("area_name_en")["log_price"].diff().shift(-1)
        monthly_df["lag_1_return"] = monthly_df.groupby("area_name_en")["future_log_return_1m"].shift(1)
        monthly_df["lag_2_return"] = monthly_df.groupby("area_name_en")["future_log_return_1m"].shift(2)

        monthly_df = monthly_df.dropna(subset=["future_log_return_1m", "lag_1_return", "lag_2_return"]).copy()

        for asset in ['oil', 'gold', 'dfm']:
            monthly_df[f"{asset}_return"] = monthly_df.groupby("area_name_en")[f"{asset}_close"].transform("pct_change")
            monthly_df[f"{asset}_return_3m"] = monthly_df.groupby("area_name_en")[f"{asset}_return"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            monthly_df[f"{asset}_vol_3m"] = monthly_df.groupby("area_name_en")[f"{asset}_return"].transform(lambda x: x.rolling(3, min_periods=1).std())

        monthly_df = monthly_df.fillna(0)

        counts = monthly_df["area_name_en"].value_counts()
        valid_areas = counts[counts >= 50].index
        monthly_df = monthly_df[monthly_df["area_name_en"].isin(valid_areas)].reset_index(drop=True)

        valid_names = set(monthly_df["area_name_en"].unique())
        source_id_map = {k: v for k, v in source_id_map.items() if k in valid_names}

        logger.info(f"Applying high-zero columns filter (Threshold: {self.zero_threshold * 100}%)...")
        col_to_drop = [col for col in monthly_df.columns if (monthly_df[col] == 0).mean() > self.zero_threshold]
        col_to_drop = [c for c in col_to_drop if c not in ['area_id', 'model_area_id', 'month', 'monthly_price', 'future_log_return_1m', 'area_name_en', 'actual_area_id']]
        monthly_df = monthly_df.drop(columns=col_to_drop)
        logger.info(f"Dropped {len(col_to_drop)} high-zero feature columns from matrix.")

        monthly_df = monthly_df.sort_values(['model_area_id', 'month'])
        return monthly_df, source_id_map, model_id_map

    def _prepare_chronos_grid(self, monthly_df):
        logger.info("Preparing uniform monthly grid for Chronos...")
        grid_df = monthly_df[monthly_df['month'] >= '2010-01-01'].copy()
        grid_df = grid_df.drop_duplicates(subset=['model_area_id', 'month'])
        grid_df = grid_df.set_index('month').groupby('model_area_id').resample('MS').asfreq().reset_index()
        grid_df = grid_df.sort_values(['model_area_id', 'month'])

        feature_cols = [c for c in grid_df.columns if c not in ['area_id', 'model_area_id', 'month', 'area_name_en', 'actual_area_id', 'log_price', 'future_log_return_1m']]
        for col in feature_cols:
            if col in grid_df.columns:
                grid_df[col] = grid_df.groupby('model_area_id')[col].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

        return grid_df.fillna(0), feature_cols

    def execute_dual_modeling_pipeline(self, monthly_df, force_start_month=None):
        chronos_set, dynamic_features = self._prepare_chronos_grid(monthly_df)
        final_keep_features = [c for c in self.tokeep_base if c in chronos_set.columns] + [f for f in dynamic_features if f not in self.tokeep_base]
        final_keep_features = list(dict.fromkeys(final_keep_features))

        if force_start_month:
            max_available_month = pd.to_datetime(force_start_month)
            chronos_set = chronos_set[chronos_set['month'] <= max_available_month].copy()
            logger.info(f"Custom forecast start month enforced: {max_available_month.strftime('%Y-%m')}")
        else:
            max_available_month = chronos_set['month'].max()

        backtest_cutoff = max_available_month - pd.DateOffset(months=self.backtest_periods)

        historical_df = chronos_set[['area_id', 'model_area_id', 'actual_area_id', 'month', 'monthly_price']].copy()

        logger.info(f"Executing Dual Run A: Processing validation splits at cutoff: {backtest_cutoff.strftime('%Y-%m-%d')}...")
        train_context = chronos_set[chronos_set['month'] <= backtest_cutoff].copy()
        test_ground_truth = chronos_set[chronos_set['month'] > backtest_cutoff].copy()

        test_counts = test_ground_truth['model_area_id'].value_counts()
        valid_test_areas = test_counts[test_counts == self.backtest_periods].index
        train_context = train_context[train_context['model_area_id'].isin(valid_test_areas)].copy()
        test_ground_truth = test_ground_truth[test_ground_truth['model_area_id'].isin(valid_test_areas)].copy()

        backtest_metrics = pd.DataFrame()
        if not test_ground_truth.empty:
            test_covariates = test_ground_truth.drop(columns=['monthly_price'])
            backtest_preds = self.pipeline.predict_df(
                df=train_context[final_keep_features],
                future_df=test_covariates[[c for c in final_keep_features if c != 'monthly_price']],
                prediction_length=self.backtest_periods,
                id_column="model_area_id",
                timestamp_column="month",
                target="monthly_price"
            )
            eval_df = test_ground_truth[['model_area_id', 'month', 'monthly_price']].merge(
                backtest_preds[['model_area_id', 'month', 'predictions']], on=['model_area_id', 'month'], how='inner'
            )
            metrics_store = []
            for area_id, group in eval_df.groupby('model_area_id'):
                y_true = group['monthly_price'].values
                y_pred = group['predictions'].values
                metrics_store.append({
                    'model_area_id': area_id,
                    'MSE': mean_squared_error(y_true, y_pred),
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'MAPE': mean_absolute_percentage_error(y_true, y_pred)
                })
            backtest_metrics = pd.DataFrame(metrics_store)
            logger.info("Run A (Dynamic Validation Loop) finished successfully.")

        logger.info(f"Executing Dual Run B: Initializing {self.prediction_length}-month future timeline forecasts...")
        production_preds = self.pipeline.predict_df(
            df=chronos_set[final_keep_features],
            prediction_length=self.prediction_length,
            id_column="model_area_id",
            timestamp_column="month",
            target="monthly_price"
        )

        production_forecasts = []
        for area_id, group in production_preds.groupby('model_area_id'):
            last_known_price = chronos_set[chronos_set['model_area_id'] == area_id].sort_values('month')['monthly_price'].iloc[-1]
            group = group.sort_values('month').copy()
            group['predicted_mom_growth_pct'] = group['predictions'].pct_change() * 100
            group['predicted_mom_growth_pct'] = group['predicted_mom_growth_pct'].astype('float64')
            group.iloc[0, group.columns.get_loc('predicted_mom_growth_pct')] = ((group['predictions'].iloc[0] / last_known_price) - 1) * 100
            production_forecasts.append(group)

        production_df = pd.concat(production_forecasts, ignore_index=True)
        output_forecast = production_df[['model_area_id', 'month', 'predictions', 'predicted_mom_growth_pct']].copy()
        output_forecast.rename(columns={'predictions': 'predicted_monthly_price'}, inplace=True)
        return backtest_metrics, output_forecast, historical_df


def run_entire_pipeline(trans_path, config_settings):
    pred_len = config_settings.get('prediction_points', 6)
    backtest_len = config_settings.get('backtest_periods', 6)
    start_month = config_settings.get('forecast_start_month', None)

    engine = DubaiPropertyForecaster(prediction_length=pred_len, backtest_periods=backtest_len)
    monthly_data, source_id_map, model_id_map = engine.process_and_engineer_data(trans_path)
    backtest_metrics, future_mom_forecast, historical_df = engine.execute_dual_modeling_pipeline(monthly_data, force_start_month=start_month)

    model_to_name = monthly_data[['model_area_id', 'area_name_en']].drop_duplicates().set_index('model_area_id')['area_name_en'].to_dict()
    model_to_saved = monthly_data[['model_area_id', 'area_id']].drop_duplicates().set_index('model_area_id')['area_id'].to_dict()
    model_to_actual = monthly_data[['model_area_id', 'actual_area_id']].drop_duplicates().set_index('model_area_id')['actual_area_id'].to_dict()

    future_mom_forecast['area_id'] = future_mom_forecast['model_area_id'].map(model_to_saved)
    future_mom_forecast['actual_area_id'] = future_mom_forecast['model_area_id'].map(model_to_actual)
    future_mom_forecast['area_name'] = future_mom_forecast['model_area_id'].map(model_to_name)

    historical_df['area_id'] = historical_df['model_area_id'].map(model_to_saved)
    historical_df['actual_area_id'] = historical_df['model_area_id'].map(model_to_actual)
    historical_df['area_name'] = historical_df['model_area_id'].map(model_to_name)

    if not backtest_metrics.empty:
        backtest_metrics['area_id'] = backtest_metrics['model_area_id'].map(model_to_saved)
        backtest_metrics['actual_area_id'] = backtest_metrics['model_area_id'].map(model_to_actual)
        backtest_metrics['area_name'] = backtest_metrics['model_area_id'].map(model_to_name)

    return backtest_metrics, future_mom_forecast, historical_df


def execute_pipeline_entry(config):
    logger.info("Starting Forecasting Stage Orchestration...")
    base_dir = config['paths']['base_dir']
    input_path = os.path.join(base_dir, config['paths']['chronos_input'])
    output_path = os.path.join(base_dir, config['paths']['chronos_output'])
    backtest_path = os.path.join(base_dir, config['paths']['chronos_backtest_output'])
    historic_path = os.path.join(base_dir, config['paths']['chronos_historic_output'])
    old_dir = os.path.join(base_dir, config['paths'].get('old_files_dir', 'old_files'))
    os.makedirs(old_dir, exist_ok=True)

    forecast_settings = config.get('forecasting_settings', {})
    prefix = config.get('archive', {}).get('prefix', 'old_')

    archive_existing_file(output_path, old_dir, prefix)
    archive_existing_file(backtest_path, old_dir, prefix)
    archive_existing_file(historic_path, old_dir, prefix)

    logger.info(f"Target Input File: {input_path}")
    backtest, forecast, historic = run_entire_pipeline(input_path, forecast_settings)

    forecast.to_csv(output_path, index=False)
    historic.to_csv(historic_path, index=False)
    logger.info(f"✅ Forecasting engine outputs successfully saved to: {output_path}")
    logger.info(f"✅ Historical timeline successfully saved to: {historic_path}")

    if not backtest.empty:
        backtest.to_csv(backtest_path, index=False)
        logger.info(f"✅ Backtest metrics successfully saved to: {backtest_path}")
    else:
        logger.warning("⚠️ No backtest metrics were generated to save.")

    return backtest, forecast, historic