import pandas as pd
import numpy as np
import os
import time
import yaml
import logging
from dagshub import get_repo_bucket_client

# Initialize DagsHub client globally
fs = get_repo_bucket_client("poojariprakash88/truestates-ml-ops", flavor="s3fs")


# ---------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# OUTLIER TREATMENT CONFIGURATION & FUNCTIONS
# ---------------------------------------------------------
room_to_id = {
    'Studio': 1, '1 B/R': 2, '2 B/R': 3, '3 B/R': 4,
    '4 B/R': 5, '5 B/R': 6, '6 B/R': 7, '7 B/R': 8, '8 B/R': 9
}

min_area_by_subtype = {
    1: 25, 2: 50, 3: 80, 4: 100, 5: 140,
    6: 170, 7: 200, 8: 250, 9: 300, 10: 330, 621: 80
}

def extract_floor_value(df, source_col='floor_key', target_col='floor'):
    logger.info("🧹 Extracting numeric floor values...")
    is_missing = df[source_col].isna() | (df[source_col].astype(str).str.strip() == '')
    extracted_nums = df[source_col].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df[target_col] = extracted_nums
    df[target_col] = df[target_col].fillna(0.5)
    df.loc[is_missing, target_col] = 0.0
    return df

def clean_area_with_hard_min(df):
    logger.info("🧹 Step 1: Global Area Outlier Treatment")
    room_col = 'rooms_en'
    if room_col not in df.columns or 'procedure_area' not in df.columns:
        return df

    cleaned_list = []
    room_groups = df.groupby(room_col)

    for room, group in room_groups:
        room_id = room_to_id.get(room, 0)
        lower_limit = min_area_by_subtype.get(room_id, group['procedure_area'].quantile(0.05))
        upper_limit = group['procedure_area'].quantile(0.99)
        filtered_group = group[(group['procedure_area'] >= lower_limit) &
                               (group['procedure_area'] <= upper_limit)]
        cleaned_list.append(filtered_group)

    if not cleaned_list: return df

    cleaned_df = pd.concat(cleaned_list)
    logger.info(f"✅ Removed {len(df) - len(cleaned_df)} rows due to invalid 'procedure_area'.")
    return cleaned_df

def remove_granular_outliers(df, price_col='meter_sale_price'):
    logger.info("🧹 Step 2: Granular Price Outlier Treatment")
    if 'area_name_en' not in df.columns or price_col not in df.columns:
        return df

    group_counts = df.groupby('area_name_en')[price_col].transform('count')
    lower_bounds = df.groupby('area_name_en')[price_col].transform(lambda x: x.quantile(0.01))
    upper_bounds = df.groupby('area_name_en')[price_col].transform(lambda x: x.quantile(0.99))
    
    mask = (group_counts < 10) | ((df[price_col] >= lower_bounds) & (df[price_col] <= upper_bounds))
    cleaned_df = df[mask].copy()
    logger.info(f"✅ Removed {len(df) - len(cleaned_df)} rows due to extreme '{price_col}'.")
    return cleaned_df

def apply_all_outlier_treatments(df, output_dir='.'):
    logger.info("📊 METADATA: BEFORE OUTLIER TREATMENT 📊")
    stats_before = pd.DataFrame()
    if 'meter_sale_price' in df.columns and 'area_name_en' in df.columns:
        stats_before = df.groupby('area_name_en')['meter_sale_price'].describe().add_suffix('_before')

    initial_len = len(df)
    df_cleaned = clean_area_with_hard_min(df)
    df_cleaned = remove_granular_outliers(df_cleaned, price_col='meter_sale_price')
    
    logger.info("📊 METADATA: AFTER OUTLIER TREATMENT 📊")
    stats_after = pd.DataFrame()
    if 'meter_sale_price' in df_cleaned.columns and 'area_name_en' in df_cleaned.columns:
        stats_after = df_cleaned.groupby('area_name_en')['meter_sale_price'].describe().add_suffix('_after')

    if not stats_before.empty or not stats_after.empty:
        try:
            metadata_df = stats_before.join(stats_after, how='outer').reset_index()
            csv_path = os.path.join(output_dir, 'outliers_treatment_metadata.csv')
            metadata_df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"❌ Failed to save metadata CSV. Error: {e}")

    logger.info(f"🚀 TOTAL OUTLIERS REMOVED: {initial_len - len(df_cleaned)}")
    return df_cleaned


# ---------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------------------------
def process_and_merge_transactions(trans_path, micro_path, output_path, proj_grade_path, 
                                   dev_grade_path, bld_grade_path, projects_path, developers_path, data_dir="."):
    start_proc = time.time()
    logger.info("Loading main transaction data...")

    # 1. Load Transactions Data
    with fs.open(trans_path, "rb") as f:
        df = pd.read_parquet(f)

    # 2. Filter Out Invalid Rooms
    room_col = 'rooms_en' 
    if room_col in df.columns:
        exclude_rooms = ['Office', 'Shop', 'Single Room', '0', 'No_rooms_en', 'Missing_rooms_en']
        df = df[~df[room_col].isin(exclude_rooms)]

    # 3. Apply Outlier Treatments
    final_out_dir = os.path.dirname(output_path)
    os.makedirs(final_out_dir, exist_ok=True) 
    df = apply_all_outlier_treatments(df, output_dir=final_out_dir)

    # 4. Generate trans_ubp_key
    sub_type = pd.to_numeric(df.get('property_sub_type_id', 0), errors='coerce').fillna(0).astype(int).astype(str)
    area_id = pd.to_numeric(df.get('area_id', 0), errors='coerce').fillna(0).astype(int).astype(str)
    proj_num = pd.to_numeric(df.get('project_number', 0), errors='coerce').fillna(0).astype(int).astype(str)
    proc_area = pd.to_numeric(df.get('procedure_area', 0), errors='coerce').fillna(0).astype(int).astype(str)
    rooms = df.get('rooms_en', pd.Series([''] * len(df))).fillna('').astype(str).str.strip()
    df['trans_ubp_key'] = sub_type + '-' + area_id + '-' + proj_num + '-' + rooms + '-' + proc_area

    # 5. Load & Clean Micro Data
    with fs.open(micro_path, "rb") as f:
        micro_df = pd.read_csv(f, low_memory=False)
    micro_df = micro_df.loc[:, ~micro_df.columns.str.endswith('_y')]
    micro_df.columns = [c[:-2] if c.endswith('_x') else c for c in micro_df.columns]
    
    micro_drop = ['Unnamed: 0', 'flats', 'unit_number', 'rooms', 'unit_parking_number', 
                  'car_parks', 'shops', 'offices', 'project_status', 'no_of_lands']
    micro_df.drop(columns=[c for c in micro_drop if c in micro_df.columns], inplace=True)
    if 'trans_ubp_key' in micro_df.columns:
        micro_df.drop_duplicates(subset=['trans_ubp_key'], inplace=True)

    # ---------------------------------------------------------
    # 6. Merge Cleaned Transactions with Micro Data
    # ---------------------------------------------------------
    logger.info("Merging Transactions with Micro Metadata...")
    if 'trans_ubp_key' in df.columns and 'trans_ubp_key' in micro_df.columns:
        cols_before = set(df.columns)
        
        df = df.merge(micro_df, on='trans_ubp_key', how='left')
        df = df.loc[:, ~df.columns.str.endswith('_y')]
        df.columns = [c[:-2] if c.endswith('_x') else c for c in df.columns]
        
        final_drops = ['trans_ubp_key', 'cancellation_date', 'rooms_en_bld', 'actual_common_area', 
                       'common_area', 'percent_completed', 'is_lease_hold', 'is_free_hold', 'Old/New Dubai']
        df.drop(columns=final_drops, inplace=True, errors='ignore')
        
        cols_after = set(df.columns)
        added_cols = list(cols_after - cols_before)
        logger.info(f"✅ Micro Data Merge Complete. Added {len(added_cols)} columns: {added_cols}")

    # ---------------------------------------------------------
    # 7. Enrich Missing Developer Info (Projects & Developers Parquet)
    # ---------------------------------------------------------
    logger.info("Enriching missing Developer Information from Parquet files...")
    if os.path.exists(projects_path) and os.path.exists(developers_path):
        cols_before_enrichment = set(df.columns)
        
        # Helper function to count missing values accurately
        def get_missing_count(series, placeholder_text):
            if series is None: return len(df)
            return series.isna().sum() + (series.astype(str).str.strip() == '').sum() + (series.astype(str) == placeholder_text).sum()

        missing_dev_num_before = get_missing_count(df.get('developer_number'), 'Missing_developer_number')
        missing_dev_name_before = get_missing_count(df.get('developer_name_en'), 'Missing_developer_name_en')
        
        logger.info(f"📊 BEFORE ENRICHMENT:")
        logger.info(f"   -> Missing developer_number : {missing_dev_num_before}")
        logger.info(f"   -> Missing developer_name_en: {missing_dev_name_before}")

        # Load Parquet Files
        with fs.open(projects_path, "rb") as f:
            projects_df = pd.read_parquet(f)
        with fs.open(developers_path, "rb") as f:
            developers_df = pd.read_parquet(f)
        
        # 1. Merge Projects to get developer_number
        if 'project_number' in df.columns and 'project_number' in projects_df.columns:
            proj_subset = projects_df[['project_number', 'developer_number']].drop_duplicates()
            df = pd.merge(df, proj_subset, on='project_number', how='left', suffixes=('_x', '_y'))
            
            if 'developer_number_y' in df.columns and 'developer_number_x' in df.columns:
                df['developer_number_x'] = df['developer_number_x'].fillna(df['developer_number_y'])
                df.rename(columns={'developer_number_x': 'developer_number'}, inplace=True)
                df.drop(columns=['developer_number_y'], inplace=True)
            elif 'developer_number_y' in df.columns:
                df.rename(columns={'developer_number_y': 'developer_number'}, inplace=True)

        # 2. Merge Developers to get developer_name_en
        dev_subset = developers_df[['developer_number', 'developer_name_en']].drop_duplicates()
        if 'developer_number' in df.columns:
            if 'developer_name_en' in df.columns:
                df['developer_name_en'] = df['developer_name_en'].replace('Missing_developer_name_en', np.nan)
                
            df = pd.merge(df, dev_subset, on='developer_number', how='left', suffixes=('_x', '_y'))
            
            if 'developer_name_en_y' in df.columns and 'developer_name_en_x' in df.columns:
                df['developer_name_en_x'] = df['developer_name_en_x'].fillna(df['developer_name_en_y'])
                df.rename(columns={'developer_name_en_x': 'developer_name_en'}, inplace=True)
                df.drop(columns=['developer_name_en_y'], inplace=True)
            elif 'developer_name_en_y' in df.columns:
                df.rename(columns={'developer_name_en_y': 'developer_name_en'}, inplace=True)

        # 3. Final Fallback Mapping (Ensure all existing matches map internally)
        if 'developer_name_en' in df.columns and 'developer_number' in df.columns:
            valid_devs = df.dropna(subset=['developer_name_en'])
            dev_mapping = valid_devs.drop_duplicates(subset=['developer_number']).set_index('developer_number')['developer_name_en'].to_dict()
            df['developer_name_en'] = df['developer_name_en'].fillna(df['developer_number'].map(dev_mapping))
            df['developer_name_en'] = df['developer_name_en'].fillna('Missing_developer_name_en')

        # Calculate AFTER metrics
        missing_dev_num_after = get_missing_count(df.get('developer_number'), 'Missing_developer_number')
        missing_dev_name_after = get_missing_count(df.get('developer_name_en'), 'Missing_developer_name_en')
        
        logger.info(f"📊 AFTER ENRICHMENT:")
        logger.info(f"   -> Missing developer_number : {missing_dev_num_after} (Recovered: {missing_dev_num_before - missing_dev_num_after})")
        logger.info(f"   -> Missing developer_name_en: {missing_dev_name_after} (Recovered: {missing_dev_name_before - missing_dev_name_after})")
        
        cols_after_enrichment = set(df.columns)
        added_cols = list(cols_after_enrichment - cols_before_enrichment)
        logger.info(f"✅ Developer Enrichment Complete. Added columns: {added_cols if added_cols else 'None (Values Updated)'}")
    else:
        logger.warning(f"⚠️ Developers or Projects parquet not found in path. Skipping enrichment.")

    # ---------------------------------------------------------
    # 8. Rule-based Developer Mapping (Catch-all)
    # ---------------------------------------------------------
    dev_rules = {
        "emaar": "Emaar Properties", "emirates living": "Emaar Properties", "springs": "Emaar Properties",
        "binghatti": "Binghatti Developers", "damac": "DAMAC Properties", "sobha": "Sobha Realty",
        "nakheel": "Nakheel Properties", "dubai properties": "Dubai Properties", "meraas": "Meraas",
        "habtoor": "Habtoor Group", "ellington": "Ellington Properties", "omniyat": "Omniyat",
        "aldar": "Aldar Properties", "majid": "Majid Al Futtaim", "mag": "MAG Property Development"
    }

    if 'developer_name_en' in df.columns:
        missing_dev = df['developer_name_en'].isna() | (df['developer_name_en'] == "") | (df['developer_name_en'] == 'Missing_developer_name_en')
        for pattern, replacement in dev_rules.items():
            mask = missing_dev & df['project_name_en'].astype(str).str.contains(pattern, case=False, na=False)
            df.loc[mask, 'developer_name_en'] = replacement

    # ---------------------------------------------------------
    # 9. Merge Building, Developer & Project Scorecards 
    # ---------------------------------------------------------
    logger.info("Merging Excel Scorecards...")
    df['building_name_en'] = df['building_name_en'].astype(str).str.strip()
    df['developer_name_en'] = df['developer_name_en'].astype(str).str.strip()
    df['project_name_en'] = df['project_name_en'].astype(str).str.strip()

    # A. Merge Building Data
    if fs.exists(bld_grade_path):
        cols_before = set(df.columns)
        
        with fs.open(bld_grade_path, "rb") as f:
            bld_df = pd.read_excel(f, sheet_name='BLD_class')
            
        bld_df['Building Name'] = bld_df['Building Name'].astype(str).str.strip()
        
        bld_cols = ['Building Name', 'Price/sqft (AED)', 'Locality Zone', 'Score', 'Grade', 'Price Tier', 'Reputation']
        bld_cols = [c for c in bld_cols if c in bld_df.columns] 
        
        df = pd.merge(df, bld_df[bld_cols], left_on='building_name_en', right_on='Building Name', how='left')
        
        # Clean numeric fields with coerce, then fillna
        if 'Grade' in df.columns: df['Grade'] = df['Grade'].fillna('Unknown')
        if 'Score' in df.columns: df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
        if 'Price Tier' in df.columns: df['Price Tier'] = df['Price Tier'].fillna('Unknown')
        if 'Reputation' in df.columns: df['Reputation'] = df['Reputation'].fillna('Unknown')
        if 'Price/sqft (AED)' in df.columns: df['Price/sqft (AED)'] = pd.to_numeric(df['Price/sqft (AED)'], errors='coerce')
        
        added_cols = list(set(df.columns) - cols_before)
        logger.info(f"✅ Building Scorecard Merge Complete. Added columns: {added_cols}")
    else:
        logger.warning(f"⚠️ MISSING FILE: Could not find Building scorecard at {bld_grade_path}")

    # B. Merge Developer Scorecard
    if fs.exists(dev_grade_path):
        cols_before = set(df.columns)
        
        with fs.open(dev_grade_path, "rb") as f:
            dev_df = pd.read_excel(f, sheet_name='developer_scorecard')
            
        dev_df['Developer Name'] = dev_df['Developer Name'].astype(str).str.strip()
        
        dev_cols = ['Developer Name', 'Developer Score (0-100)', 'Grade', 'Developer Tier']
        dev_subset = dev_df[dev_cols].rename(columns={'Grade': 'Developer_grade'})
        
        df = pd.merge(df, dev_subset, left_on='developer_name_en', right_on='Developer Name', how='left')
        
        # Clean numeric fields with coerce, then fillna
        if 'Developer_grade' in df.columns: df['Developer_grade'] = df['Developer_grade'].fillna('Unknown')
        if 'Developer Score (0-100)' in df.columns: df['Developer Score (0-100)'] = pd.to_numeric(df['Developer Score (0-100)'], errors='coerce').fillna(0)
        if 'Developer Tier' in df.columns: df['Developer Tier'] = df['Developer Tier'].fillna('Unknown')
        
        added_cols = list(set(df.columns) - cols_before)
        logger.info(f"✅ Developer Scorecard Merge Complete. Added columns: {added_cols}")
    else:
        logger.warning(f"⚠️ MISSING FILE: Could not find Developer scorecard at {dev_grade_path}")

    # C. Merge Project Scorecard
    if fs.exists(proj_grade_path):
        cols_before = set(df.columns)
        
        with fs.open(proj_grade_path, "rb") as f:
            proj_df = pd.read_excel(f, sheet_name='Scored Projects')
            
        proj_df['Project Name'] = proj_df['Project Name'].astype(str).str.strip()
        
        proj_cols = ['Project Name', 'Developer Reputation Tier', 'Location / Positioning Tier', 
                     'Est. Gross Rental Yield (%)', 'Composite Score (0-100)', 'Grade']
        proj_subset = proj_df[proj_cols].rename(columns={'Grade': 'project_grade'})
        
        df = pd.merge(df, proj_subset, left_on='project_name_en', right_on='Project Name', how='left')
        
        # Clean numeric fields with coerce, then fillna
        if 'project_grade' in df.columns: df['project_grade'] = df['project_grade'].fillna('Unknown')
        if 'Composite Score (0-100)' in df.columns: df['Composite Score (0-100)'] = pd.to_numeric(df['Composite Score (0-100)'], errors='coerce').fillna(0)
        if 'Developer Reputation Tier' in df.columns: df['Developer Reputation Tier'] = df['Developer Reputation Tier'].fillna('Unknown')
        if 'Location / Positioning Tier' in df.columns: df['Location / Positioning Tier'] = df['Location / Positioning Tier'].fillna('Unknown')
        if 'Est. Gross Rental Yield (%)' in df.columns: df['Est. Gross Rental Yield (%)'] = pd.to_numeric(df['Est. Gross Rental Yield (%)'], errors='coerce').fillna(0)
        
        added_cols = list(set(df.columns) - cols_before)
        logger.info(f"✅ Project Scorecard Merge Complete. Added columns: {added_cols}")
    else:
        logger.warning(f"⚠️ MISSING FILE: Could not find Project scorecard at {proj_grade_path}")

    # ---------------------------------------------------------
    # 10. Amenities & Metro Logic
    # ---------------------------------------------------------
    if 'swimming_pools' in df.columns:
        df['swimming_pool'] = (df['swimming_pools'].fillna(0) != 0).astype(int)
    if 'unit_balcony_area' in df.columns:
        df['balcony'] = (df['unit_balcony_area'].fillna(0) != 0).astype(int)
    if 'elevators' in df.columns:
        df['elevator'] = (df['elevators'].fillna(0) != 0).astype(int)
    if 'nearest_metro_en' in df.columns:
        invalid_list = ['no_nearest_metro_en', 'missing_nearest_metro_en', 'nan', 'none']
        no_metro_mask = (df['nearest_metro_en'].astype(str).str.lower().isin(invalid_list)) | (df['nearest_metro_en'].isna())
        df['metro'] = (~no_metro_mask).astype(int)

    # ---------------------------------------------------------
    # 11. Imputation & Cleanup
    # ---------------------------------------------------------
    room_map = {
        '4 B/R': 'More than 3B/R', '5 B/R': 'More than 3B/R', '6 B/R': 'More than 3B/R', 
        '7 B/R': 'More than 3B/R', '8 B/R': 'More than 3B/R', '9 B/R': 'More than 3B/R', '10 B/R': 'More than 3B/R'
    }
    room_col = 'rooms_en_u' if 'rooms_en_u' in df.columns else 'rooms_en'
    if room_col in df.columns: df[room_col] = df[room_col].replace(room_map)

    fill_targets = ['elevators', 'swimming_pools', 'land_type_en', 'nearest_mall_en', 
                    'nearest_metro_en', 'nearest_landmark_en', 'project_start_date', 
                    'project_end_date', 'completion_date', 'built_up_area']
    
    for col in fill_targets:
        if col in df.columns:
            group_col = 'building_name_en' if col in ['elevators', 'swimming_pools'] else 'project_name_en'
            df[col] = df[col].fillna(df.groupby(group_col)[col].transform('first'))

    if 'unit_balcony_area' in df.columns:
        df.loc[(df['property_type_en'].isin(['Building', 'Villa'])) & 
               (df['unit_balcony_area'].isna()), 'unit_balcony_area'] = 0
               
    num_cols = ['has_parking', 'meter_sale_price', 'unit_balcony_area', 'built_up_area', 'swimming_pools']
    num_cols = [c for c in num_cols if c in df.columns]
    if num_cols: df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df = extract_floor_value(df, source_col='floor_key', target_col='floor')
    
    cat_cols = ['project_name_en', 'land_type_en', 'developer_name_en', 
                'building_name_en', 'property_sub_type_en', 'floor_key']
    for col in cat_cols:
        if col in df.columns: df[col] = df[col].fillna(f"Missing_{col}")

    # ---------------------------------------------------------
    # 12. Save Output
    # ---------------------------------------------------------
    logger.info(f"Saving final file to {output_path}...")
    with fs.open(output_path, "wb") as f:
        df.to_parquet(f, index=False, engine='pyarrow')
    logger.info(f"✅ Final file saved successfully. Shape: {df.shape} | Time: {time.time() - start_proc:.2f}s")
    
    return df

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Grab base directory and strip 's3://'
    base = config['paths']['base_dir'].replace("s3://", "")
    
    # Attach base to all file paths natively
    for key, value in config['paths'].items():
        if isinstance(value, str) and (value.endswith('.csv') or value.endswith('.parquet') or value.endswith('.xlsx')):
            config['paths'][key] = f"{base}/{value}"
            
    return config

def run_merging_pipeline():
    pd.set_option('io.parquet.engine', 'pyarrow')
    try:
        config = load_config()
        
        # Load core pipeline files straight from the clean config!
        paths_dict = {
            'transactions': config['paths']['merging_input_trans'],
            'micro_metadata': config['paths']['merging_input_micro'],
            'projects_path': config['paths'].get('projects_file'),
            'developers_path': config['paths'].get('developers_file'),
            'proj_grade_path': config['paths'].get('project_grade'),
            'dev_grade_path': config['paths'].get('dev_grade'),
            'bld_grade_path': config['paths'].get('bld_grade'),
            'output': config['paths']['merging_output']
        }
    except KeyError as e:
        logger.error(f"❌ Missing key in config.yaml: {e}")
        return None

    logger.info("🚀 Starting Merge & Imputation Pipeline...")
    return process_and_merge_transactions(
        trans_path=paths_dict['transactions'], 
        micro_path=paths_dict['micro_metadata'], 
        output_path=paths_dict['output'], 
        proj_grade_path=paths_dict['proj_grade_path'], 
        dev_grade_path=paths_dict['dev_grade_path'], 
        bld_grade_path=paths_dict['bld_grade_path'], 
        projects_path=paths_dict['projects_path'], 
        developers_path=paths_dict['developers_path']
    )

if __name__ == "__main__":
    run_merging_pipeline()
