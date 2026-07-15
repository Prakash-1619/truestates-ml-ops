import pandas as pd
import os
import yaml
import time
from dagshub import get_repo_bucket_client

# Initialize DagsHub client globally
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")

# Set global engine preference
pd.set_option('io.parquet.engine', 'pyarrow')

def load_config(config_path="config.yaml"):
    """Loads configuration and constructs absolute paths."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Grab the root base directory (strips 's3://')
    base = config['paths']['base_dir'].replace("s3://", "")
    
    # Attach base to the specific files this script needs
    config['paths']['trans_in'] = f"{base}/{config['paths']['transactions_input']}"
    config['paths']['trans_out'] = f"{base}/{config['paths']['transactions_output']}"
    
    return config
    
def process_transaction_data(config):
    """
    Loads, cleans, and transforms transaction data based on config settings.
    """
    start_time = time.time()
    
    # 1. Load Data
    input_path = config['paths']['trans_in']
    print(f"⏳ Loading: {input_path}")
    
    with fs.open(input_path, "rb") as f:
        df = pd.read_parquet(f)
        
    print(f"Initial shape: {df.shape}")

    # 2. Date Conversion & Filtering
    settings = config['transaction_settings']
    
    # Convert to datetime; auto-infer format to prevent wiping out data
    df['instance_date'] = pd.to_datetime(
        df['instance_date'], 
        errors="coerce"
    )
    
    # Drop NaT values BEFORE doing the limit comparison
    df = df.dropna(subset=['instance_date'])
    
    # Ensure the config date is a proper datetime object for comparison
    start_date_limit = pd.to_datetime(settings['start_filter_date'])
    
    # Apply the filter safely
    df = df[df['instance_date'] >= start_date_limit].copy()
    
    print(f"Shape after date filtering: {df.shape}")

    # 3. Clean Arabic Columns
    ar_cols = [col for col in df.columns if col.endswith('_ar')]
    df.drop(columns=ar_cols, inplace=True, errors='ignore')

    # 4. Feature Engineering (Date Parts)
    df['year_month'] = df['instance_date'].dt.to_period("M").astype(str)
    df['year'] = df['instance_date'].dt.year
    df['month'] = df['instance_date'].dt.month
    df['day_of_week'] = df['instance_date'].dt.day_name()
    df['day_of_year'] = df['instance_date'].dt.dayofyear
    df['week_of_year'] = df['instance_date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['instance_date'].dt.quarter

    # 5. Domain Filtering using YAML settings
    if 'trans_group_en' in df.columns:
        # 1. Clean the column, safely handling NaNs
        df['trans_group_en'] = df['trans_group_en'].fillna('').astype(str).str.strip()
        
        # 2. Get the exclude value(s) from settings (default to empty list if missing)
        exclude_val = settings.get('exclude_group', [])
        
        # 3. Handle it safely whether it's a list or someone accidentally changed the YAML to a single string
        if isinstance(exclude_val, list):
            exclude_list = [str(x).strip() for x in exclude_val]
        elif isinstance(exclude_val, str) and exclude_val.strip():
            exclude_list = [exclude_val.strip()]
        else:
            exclude_list = []
            
        # 4. Apply the filter if we have items to exclude
        if exclude_list:
            df = df[~df['trans_group_en'].isin(exclude_list)]
    usage = str(settings.get('property_usage', '')).strip()
    p_type = str(settings.get('property_type', '')).strip()

    if 'property_usage_en' in df.columns and 'property_type_en' in df.columns:
        # Strip whitespaces to guarantee string matches don't fail silently
        df['property_usage_en'] = df['property_usage_en'].astype(str).str.strip()
        df['property_type_en'] = df['property_type_en'].astype(str).str.strip()
        
        df_res_unit = df[
            (df['property_usage_en'] == usage) & 
            (df['property_type_en'] == p_type)
        ].copy()
    else:
        df_res_unit = df.copy()
        
    print(f"Shape after usage/type filters: {df_res_unit.shape}")

    # 6. Export as PARQUET
    output_path = config['paths']['trans_out']
    
    with fs.open(output_path, "wb") as f:
        df_res_unit.to_parquet(f, engine='pyarrow', index=False)
    
    duration = time.time() - start_time
    print(f"✅ Processed {len(df_res_unit)} rows in {duration:.2f}s")
    print(f"📁 Saved to: {output_path}")
    
    return df_res_unit

def run_transaction_processing():
    """Wrapper function to be called by main.py"""
    try:
        config = load_config()
        return process_transaction_data(config)
    except Exception as e:
        print(f"❌ Error in transaction processing: {e}")
        return None

if __name__ == "__main__":
    run_transaction_processing()
