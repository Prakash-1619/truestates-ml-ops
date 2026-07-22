import logging
import random
import sys
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from price_predictor_pipeline import predict_property_price

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = CODE_DIR / "config.yaml"


def load_config():
    import yaml
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


config = load_config()
BASE_PATH = Path(config["paths"]["base_dir"]).resolve()
COMBINATIONS_CSV = BASE_PATH / "V_2.3_combinations.csv"
INPUT_RANGES_CSV = BASE_PATH / "input_ranges.csv"
FORECAST_PATH = BASE_PATH / "forecast_df_v_2_3.csv"
HISTORIC_PATH = BASE_PATH / "historic_df_v_2_3.csv"
NEWS_PATH = BASE_PATH / "news_preds_v3_fixed.csv"
OLD_DIR = BASE_PATH / config["paths"].get("old_files_dir", "old_files")
OLD_DIR.mkdir(parents=True, exist_ok=True)

PIVOT_DATE = "2026-01-31"
EARTH_RADIUS_KM = 6371.0

app = FastAPI(
    title="TruEstates API v2.3",
    description="Dubai Real Estate API with regression + forecast integration",
    version="2.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AREA_COORDS = {
    "Al Barsha First": (25.1150, 55.2050),
    "Al Barshaa South Second": (25.0921, 55.2410),
    "Al Barshaa South Third": (25.0760, 55.2150),
    "Al Barsha South Fourth": (25.0610, 55.2360),
    "Al Barsha South Fifth": (25.0485, 55.2341),
    "Al Hebiah First": (25.0334, 55.2205),
    "Al Hebiah Second": (25.0420, 55.2450),
    "Al Hebiah Third": (25.0160, 55.2380),
    "Al Hebiah Fourth": (25.0210, 55.2150),
    "Al Hebiah Sixth": (24.9950, 55.2320),
    "Al Khairan First": (25.1850, 55.3550),
    "Al Kifaf": (25.2350, 55.2950),
    "Al Merkadh": (25.1480, 55.3050),
    "Al Thanyah Third": (25.0850, 55.1550),
    "Al Thanyah Fifth": (25.0685, 55.1450),
    "Al Warsan First": (25.1650, 55.4150),
    "Al Yelayiss 2": (24.9650, 55.2650),
    "Bukadra": (25.1850, 55.3350),
    "Burj Khalifa": (25.1972, 55.2744),
    "Business Bay": (25.1850, 55.2750),
    "Hadaeq Sheikh Mohammed Bin Rashid": (25.1150, 55.2950),
    "Jabal Ali": (25.0000, 55.0500),
    "Jabal Ali First": (25.0220, 55.1050),
    "Jabal Ali Industrial Second": (24.9850, 55.1250),
    "Madinat Al Mataar": (24.8950, 55.1550),
    "Madinat Dubai Almelaheyah": (25.2650, 55.2750),
    "Madinat Hind 4": (25.0250, 55.4550),
    "Marsa Dubai": (25.0780, 55.1350),
    "Me'Aisem First": (25.0350, 55.1950),
    "Nadd Hessa": (25.1250, 55.3850),
    "Palm Deira": (25.3150, 55.3050),
    "Palm Jumeirah": (25.1124, 55.1390),
    "Ras Al Khor Industrial First": (25.1950, 55.3650),
    "Wadi Al Safa 2": (25.1200, 55.3700),
    "Wadi Al Safa 3": (25.0850, 55.3250),
    "Wadi Al Safa 4": (25.1450, 55.3050),
    "Wadi Al Safa 5": (25.0950, 55.3650),
    "Warsan Fourth": (25.1550, 55.4250),
    "Zaabeel First": (25.2200, 55.2850),
    "Zaabeel Second": (25.2050, 55.2950),
}


def load_csv_with_fallback(primary_path: Path, parse_month=True):
    try:
        df = pd.read_csv(primary_path)
    except Exception as e:
        logger.warning(f"Primary load failed for {primary_path.name}: {e}")
        fallback_path = OLD_DIR / f"old_{primary_path.name}"
        if fallback_path.exists():
            df = pd.read_csv(fallback_path)
        else:
            return pd.DataFrame()
    if parse_month and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], dayfirst=True, errors="coerce")
    return df


try:
    combinations_df = pd.read_csv(COMBINATIONS_CSV)
    combinations_df["area_name_en"] = combinations_df["area_name_en"].astype(str).str.strip()
except Exception as e:
    logger.warning(f"Combinations load failed: {e}")
    combinations_df = pd.DataFrame()

try:
    input_ranges_df = pd.read_csv(INPUT_RANGES_CSV)
    input_ranges_df["area_name_en"] = input_ranges_df["area_name_en"].astype(str).str.strip()
except Exception as e:
    logger.error(f"Input ranges load failed: {e}")
    input_ranges_df = pd.DataFrame()

if not combinations_df.empty and "area_name_en" in combinations_df.columns:
    VALID_AREAS = sorted(combinations_df["area_name_en"].dropna().unique().tolist())
elif not input_ranges_df.empty and "area_name_en" in input_ranges_df.columns:
    VALID_AREAS = sorted(input_ranges_df["area_name_en"].dropna().unique().tolist())
else:
    VALID_AREAS = []

forecast_df = load_csv_with_fallback(FORECAST_PATH, parse_month=True)
historic_df = load_csv_with_fallback(HISTORIC_PATH, parse_month=True)
news_df = load_csv_with_fallback(NEWS_PATH, parse_month=True)

if not news_df.empty and "area_name_en" in news_df.columns:
    news_df["area_name_en"] = news_df["area_name_en"].astype(str).str.strip()

logger.info(f"Startup: {len(VALID_AREAS)} areas loaded.")


def haversine_km(lat1, lon1, lat2, lon2):
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def map_latlon_to_area(lat: float, lon: float) -> str:
    distances = []
    for area, (alat, alon) in AREA_COORDS.items():
        dist = haversine_km(lat, lon, alat, alon)
        distances.append((dist, area))
    distances.sort(key=lambda x: x[0])
    return distances[0][1]


def apply_historic_clamp(res_df: pd.DataFrame, pivot_date: str):
    if res_df.empty or "month" not in res_df.columns or "median_price" not in res_df.columns:
        return res_df
    pivot_dt = pd.to_datetime(pivot_date)
    hist_part = res_df[res_df["month"] <= pivot_dt].sort_values("month").reset_index(drop=True)
    if len(hist_part) < 2:
        return res_df
    price_last = hist_part.iloc[-1]["median_price"]
    price_prev = hist_part.iloc[-2]["median_price"]
    if price_prev == 0:
        return res_df
    perc_change = (price_last - price_prev) / price_prev
    if abs(perc_change) > 0.05:
        clamp_val = random.uniform(0.035, 0.05)
        direction = 1 if perc_change > 0 else -1
        new_price = price_prev * (1 + direction * clamp_val)
        scaling_factor = new_price / price_last
        target_date = hist_part.iloc[-1]["month"]
        res_df.loc[res_df["month"] >= target_date, "median_price"] *= scaling_factor
    return res_df


def get_ref_row(area_name: str):
    if not input_ranges_df.empty and "area_name_en" in input_ranges_df.columns and area_name in input_ranges_df["area_name_en"].values:
        return input_ranges_df[input_ranges_df["area_name_en"] == area_name].iloc[0]
    if not combinations_df.empty and "area_name_en" in combinations_df.columns and area_name in combinations_df["area_name_en"].values:
        return combinations_df[combinations_df["area_name_en"] == area_name].iloc[0]
    raise ValueError(f"Area {area_name} not found in input files.")


def get_validated_payload(area_name: str, params: dict) -> dict:
    ref_row = get_ref_row(area_name)

    def resolve(key, castfunc):
        val = params.get(key)
        if val is not None:
            try:
                return castfunc(val)
            except Exception:
                pass
        return castfunc(ref_row.get(key)) if key in ref_row else None

    return {
        "area_name": area_name,
        "trans_group_en": resolve("trans_group_en", str) or "Sales",
        "rooms_en": resolve("rooms_en", str) or "Unknown",
        "reg_type_en": resolve("reg_type_en", str) or "Unknown",
        "floor_bin": resolve("floor_bin", str) or "Unknown",
        "Grade": resolve("Grade", str) or "Unknown",
        "project_grade": resolve("project_grade", str) or "Unknown",
        "Developer_grade": resolve("Developer_grade", str) or "Unknown",
        "Developer Reputation Tier": resolve("Developer Reputation Tier", str) or "Unknown",
        "Locality Zone": resolve("Locality Zone", str) or "Unknown",
        "Price Tier": resolve("Price Tier", str) or "Unknown",
        "Developer Tier": resolve("Developer Tier", str) or "Unknown",
        "Reputation": resolve("Reputation", str) or "Unknown",
        "has_parking": resolve("has_parking", int) or 0,
        "swimming_pool": resolve("swimming_pool", int) or 0,
        "balcony": resolve("balcony", int) or 0,
        "elevators": resolve("elevators", int) or 0,
        "metro": resolve("metro", int) or 0,
        "procedure_area": resolve("procedure_area", float) or 0.0,
        "Score": resolve("Score", float) or 0.0,
        "year": resolve("year", int) or pd.Timestamp.now().year,
        "month": resolve("month", int) or pd.Timestamp.now().month,
    }


@app.get("/areas")
async def list_areas():
    return {"areas": VALID_AREAS, "count": len(VALID_AREAS), "coords_available": list(AREA_COORDS.keys())}


@app.get("/areas/{area_name}")
async def area_info(area_name: str):
    if area_name not in VALID_AREAS:
        raise HTTPException(status_code=404, detail="Area not found")
    row = get_ref_row(area_name)
    coords = AREA_COORDS.get(area_name, {"lat": None, "lon": None})
    return {"area_name": area_name, "default_combination": row.to_dict(), "coordinates": coords}

@app.get("/forecast")
async def unified_forecast(
    area_name: Optional[str] = Query(None, description="Area Name"),
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude"),
    trans_group_en: Optional[str] = Query(None),
    rooms_en: Optional[str] = Query(None),
    reg_type_en: Optional[str] = Query(None),
    floor_bin: Optional[str] = Query(None),
    Grade: Optional[str] = Query(None),
    project_grade: Optional[str] = Query(None),
    Developer_grade: Optional[str] = Query(None),
    Developer_Reputation_Tier: Optional[str] = Query(None),
    Locality_Zone: Optional[str] = Query(None),
    Price_Tier: Optional[str] = Query(None),
    Developer_Tier: Optional[str] = Query(None),
    Reputation: Optional[str] = Query(None),
    has_parking: Optional[int] = Query(None, ge=0, le=1),
    swimming_pool: Optional[int] = Query(None, ge=0, le=1),
    balcony: Optional[int] = Query(None, ge=0, le=1),
    elevators: Optional[int] = Query(None, ge=0, le=1),
    metro: Optional[int] = Query(None, ge=0, le=1),
    procedure_area: Optional[float] = Query(None),
    Score: Optional[float] = Query(None),
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None),
):
    try:
        resolved_area = area_name or (map_latlon_to_area(lat, lon) if lat is not None and lon is not None else None)
        if not resolved_area:
            raise ValueError("Must provide either area_name or lat/lon coordinates.")

        params = {
            "trans_group_en": trans_group_en,
            "rooms_en": rooms_en,
            "reg_type_en": reg_type_en,
            "floor_bin": floor_bin,
            "Grade": Grade,
            "project_grade": project_grade,
            "Developer_grade": Developer_grade,
            "Developer Reputation Tier": Developer_Reputation_Tier,
            "Locality Zone": Locality_Zone,
            "Price Tier": Price_Tier,
            "Developer Tier": Developer_Tier,
            "Reputation": Reputation,
            "has_parking": has_parking,
            "swimming_pool": swimming_pool,
            "balcony": balcony,
            "elevators": elevators,
            "metro": metro,
            "procedure_area": procedure_area,
            "Score": Score,
            "year": year,
            "month": month,
        }

        final_input = get_validated_payload(resolved_area, params)
        res_df = predict_property_price(final_input, forecast_df.copy(), historic_df.copy())

        if res_df.empty:
            return {
                "area_name": resolved_area,
                "news_available": False,
                "narrative": None,
                "before_prediction": [],
                "prediction_point": [],
                "forecast": [],
            }

        res_df = res_df.copy()
        res_df["month"] = pd.to_datetime(res_df["month"], errors="coerce")
        res_df = res_df.sort_values("month")

        pivot_dt = pd.to_datetime(PIVOT_DATE)
        before_df = res_df[res_df["month"] < pivot_dt].copy()
        point_df = res_df[res_df["month"] == pivot_dt].copy()
        after_df = res_df[res_df["month"] > pivot_dt].copy()

        def fmt_series(df):
            if df.empty:
                return []
            return [
                {
                    "month": r["month"].strftime("%Y-%m-%d") if pd.notna(r["month"]) else None,
                    "median_price": round(float(r["median_price"]), 2) if pd.notna(r["median_price"]) else None,
                    "area": r.get("area", resolved_area),
                }
                for _, r in df.iterrows()
            ]

        area_news = pd.DataFrame()
        narrative = None
        news_available = False

        if not news_df.empty and "area_name_en" in news_df.columns:
            area_news = news_df[news_df["area_name_en"].astype(str).str.strip() == resolved_area].copy()
            news_available = not area_news.empty
            if news_available:
                if "narrative" in area_news.columns and area_news["narrative"].notna().any():
                    narrative = str(area_news["narrative"].dropna().iloc[0])
                elif "news_narrative" in area_news.columns and area_news["news_narrative"].notna().any():
                    narrative = str(area_news["news_narrative"].dropna().iloc[0])

        return {
            "area_name": resolved_area,
            "news_available": news_available,
            "narrative": narrative,
            "before_prediction": fmt_series(before_df),
            "prediction_point": fmt_series(point_df),
            "forecast": fmt_series(after_df),
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Prediction Error")


@app.get("/")
async def healthcheck():
    return {"status": "online", "areas_supported": len(VALID_AREAS), "pivot_date": PIVOT_DATE}