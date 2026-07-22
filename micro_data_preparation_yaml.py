import os
import re
import time
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_parent(path_str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def get_processed_project_data(config):
    projects = pd.read_parquet(config['paths']['projects_file'])
    developers = pd.read_parquet(config['paths']['developers_file'])
    cols_pro = config['ingestion_columns']['projects']
    cols_dev = config['ingestion_columns']['developers']
    pd_df = pd.merge(projects[cols_pro], developers[cols_dev], on='developer_id', how='left')
    for col in config['ingestion_processing']['date_columns']:
        if col in pd_df.columns:
            pd_df[col] = pd.to_datetime(pd_df[col], errors='coerce')
    return pd_df


def get_processed_building_data(config, pd_df):
    buildings = pd.read_parquet(config['paths']['buildings_file'])
    cols_build = config['ingestion_columns']['buildings']
    pdb_df = pd.merge(buildings[cols_build], pd_df, on=['area_id', 'project_id'], how='outer')
    if 'creation_date' in pdb_df.columns:
        pdb_df['creation_date'] = pd.to_datetime(pdb_df['creation_date'], errors='coerce')
    return pdb_df


def create_floor_bin(floor):
    if pd.isna(floor):
        return 'Unknown'
    floor_str = str(floor).upper().strip()
    if floor_str in ['-', '', 'NA']:
        return 'Unknown'
    below_first_prefixes = ('G', 'B', 'M', 'P', 'GROUND', 'BASE', 'MEZZ', 'PODIUM')
    if floor_str.startswith(below_first_prefixes) or floor_str == '0':
        return 'Below 1st Floor'
    match = re.search(r'\d+', floor_str)
    if match:
        floor_num = int(match.group())
        if floor_num == 0:
            return 'Below 1st Floor'
        lower_bound = ((floor_num - 1) // 10) * 10 + 1
        upper_bound = lower_bound + 9
        return f'{lower_bound}-{upper_bound}'
    return 'Unknown'


def get_final_integrated_data(config, pdb_df):
    unit = pd.read_parquet(config['paths']['units_file'])
    removing_cols = config['ingestion_processing']['drop_columns']
    for col in removing_cols:
        if col in unit.columns:
            unit.drop(columns=col, inplace=True)
        if col in pdb_df.columns:
            pdb_df.drop(columns=col, inplace=True)
    target_cols = ['area_id', 'land_number', 'building_number', 'rooms_en', 'property_sub_type_id', 'actual_area', 'creation_date', 'master_project_en', 'master_project_ar', 'project_id', 'project_name_en']
    pdb_df = pdb_df.rename(columns={col: f'{col}_bld' for col in target_cols if col in pdb_df.columns})
    unit = unit.rename(columns={col: f'{col}_u' for col in target_cols if col in unit.columns})
    unit.drop_duplicates(keep='last', inplace=True)
    bld_id_col = 'property_id_x' if 'property_id_x' in pdb_df.columns else 'property_id'
    pdb_df.rename(columns={bld_id_col: 'property_id_bld'}, inplace=True)
    if 'property_id' in unit.columns:
        unit.rename(columns={'property_id': 'property_id_u'}, inplace=True)
    elif 'property_id_x' in unit.columns:
        unit.rename(columns={'property_id_x': 'property_id_u'}, inplace=True)
    bld_id_col = 'property_id_bld'
    mpdd = pdb_df.drop_duplicates(subset=[bld_id_col]).copy()
    unit['parent_property_id'] = unit['parent_property_id'].astype(str)
    mpdd[bld_id_col] = mpdd[bld_id_col].astype(str)
    pdbu_df2 = pd.merge(unit, mpdd, left_on='parent_property_id', right_on=bld_id_col, how='outer')
    if 'rooms_en_u' in pdbu_df2.columns:
        pdbu_df2['rooms_en_u'] = pdbu_df2['rooms_en_u'].str.replace(r'(\d+)\s*bed\s*rooms?\s*\+\s*hall', r'\1 B/R', regex=True, case=False)
        pdbu_df2['rooms_en_u'] = pdbu_df2['rooms_en_u'].str.replace(r'^(\d+)\s*(?:B/R)?\s*\+.*$', r'\1 B/R', regex=True, case=False)
        pdbu_df2['rooms_en_u'] = pdbu_df2['rooms_en_u'].str.replace(r'^(\d+)$', r'\1 B/R', regex=True)
    fill_pairs = [(f'{c}_u', f'{c}_bld') for c in target_cols]
    for target, source in fill_pairs:
        if target in pdbu_df2.columns and source in pdbu_df2.columns:
            pdbu_df2[target] = pdbu_df2[target].fillna(pdbu_df2[source])
    x_cols = [col for col in pdbu_df2.columns if col.endswith('_x')]
    for col_x in x_cols:
        col_y = col_x.replace('_x', '_y')
        col_base = col_x.replace('_x', '')
        if col_y in pdbu_df2.columns:
            pdbu_df2[col_x] = pdbu_df2[col_x].fillna(pdbu_df2[col_y])
            pdbu_df2.drop(columns=col_y, inplace=True)
            pdbu_df2.rename(columns={col_x: col_base}, inplace=True)
    for col in ['property_sub_type_id_u', 'area_id_u', 'project_number']:
        if col in pdbu_df2.columns:
            pdbu_df2[col] = pdbu_df2[col].fillna(0).astype(int)
    if 'actual_area_u' in pdbu_df2.columns:
        pdbu_df2['actual_area_u'] = pdbu_df2['actual_area_u'].fillna(0).astype(float)
    pdbu_df2['trans_ubp_key'] = (
        pdbu_df2['property_sub_type_id_u'].astype(str) + '-' +
        pdbu_df2['area_id_u'].astype(str) + '-' +
        pdbu_df2['project_number'].astype(str) + '-' +
        pdbu_df2['rooms_en_u'].astype(str) + '-' +
        pdbu_df2['actual_area_u'].astype(int).astype(str)
    )
    pdbu_df2.drop_duplicates(subset=['trans_ubp_key'], keep='first', inplace=True)
    pdbu_df2['floor_bin'] = pdbu_df2['floor_key'].apply(create_floor_bin) if 'floor_key' in pdbu_df2.columns else 'Unknown'
    columns_to_use = ['area_id_u', 'unit_balcony_area', 'floor_key', 'floor_bin', 'rooms_en_u', 'actual_area_u', 'creation_date_u', 'creation_date_bld', 'land_type_en', 'floors', 'rooms_en_bld', 'built_up_area', 'bld_levels', 'swimming_pools', 'elevators', 'project_start_date', 'project_end_date', 'completion_date', 'cancellation_date', 'no_of_lands', 'no_of_buildings', 'no_of_villas', 'no_of_units', 'developer_name_en', 'registration_date', 'developer_number', 'project_number', 'trans_ubp_key', 'property_id_bld', 'property_id_u']
    return pdbu_df2[[c for c in columns_to_use if c in pdbu_df2.columns]]


def run_ingestion(config=None):
    config = config or load_config()
    start_total = time.time()
    pd_df = get_processed_project_data(config)
    pdb_df = get_processed_building_data(config, pd_df)
    final_df = get_final_integrated_data(config, pdb_df)
    out_path = config['paths']['ingestion_output']
    ensure_parent(out_path)
    final_df.to_csv(out_path, index=False)
    print(f'Ingestion complete: {final_df.shape} in {time.time() - start_total:.2f}s')
    return final_df


if __name__ == '__main__':
    run_ingestion()
