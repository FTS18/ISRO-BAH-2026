r"""
FastAPI Microservice — AI Digital Twin of India's Climate
Production-grade REST API for climate forecast data consumers.
Exposes endpoints for: spatial forecasts, what-if simulations, anomaly alerts, and model validation metrics.

Start this server with:
    ./venv312/Scripts/uvicorn.exe src.api.main:app --reload --port 8000
Then visit: http://localhost:8000/docs for interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import xarray as xr
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.spatial_predictions import SpatialClimatePredictor, PILOT_REGIONS
from src.climate_alerts import ClimateAlertEngine

# ─────────────────────────── App Setup ───────────────────────────
app = FastAPI(
    title="India's AI Climate Digital Twin API",
    description="""
    **Production REST API** for the AI-Powered Digital Twin of India's Climate.
    Developed for ISRO Problem Statement 5 — Hackathon Submission.

    Exposes endpoints for:
    - Real-time spatial climate state queries (IMD gridded data)
    - Short-term ConvLSTM-based spatial rainfall/temperature forecasts
    - What-if climate scenario simulation (temperature/rainfall modifiers)
    - AI model validation metrics (RMSE / MAE on holdout data)
    - Automated extreme weather anomaly alerts (IMD thresholds)
    - NICES / MOSDAC data assimilation status
    """,
    version="1.0.0",
    contact={"name": "ISRO Hackathon Team", "url": "https://isro.gov.in"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Data Loading ───────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

def _load_datasets(region="Karnataka"):
    try:
        ds_rain = xr.open_dataset(os.path.join(DATA_DIR, 'IMD_Gridded_Rainfall_0.25_Real_v4.nc'), engine='netcdf4')
        ds_temp = xr.open_dataset(os.path.join(DATA_DIR, 'IMD_Gridded_MaxTemp_1.0_Real_v3.nc'), engine='netcdf4')
        
        # Slice to region
        ds_rain = predictor.slice_region(ds_rain, region)
        ds_temp = predictor.slice_region(ds_temp, region)
        return ds_rain, ds_temp
    except Exception as e:
        raise RuntimeError(f"Could not load NetCDF datasets: {e}")

predictor = SpatialClimatePredictor()
alert_engine = ClimateAlertEngine()

# ─────────────────────────── Response Models ───────────────────────────
class ForecastResponse(BaseModel):
    pilot_region: str
    variable: str
    days_ahead: int
    grid_resolution_degrees: float
    predicted_grid_mean: float
    predicted_grid_max: float
    predicted_grid_min: float
    units: str

class WhatIfResponse(BaseModel):
    pilot_region: str
    rainfall_change_pct: float
    temp_change_c: float
    modified_rainfall_mean: float
    impact_level: str
    sector_impact_agriculture: str
    sector_impact_reservoirs: str

class ValidationResponse(BaseModel):
    pilot_region: str
    variable: str
    dataset: str
    holdout_days: int
    rmse: float
    mae: float
    units: str

class AlertResponse(BaseModel):
    pilot_region: str
    alert_count: int
    alerts: list

class ConfigureRequest(BaseModel):
    """Schema for POST /api/v1/configure — external simulation configuration interface."""
    region: str = Field(default="Karnataka", description="Target pilot region name from PILOT_REGIONS.")
    variable: str = Field(default="rainfall", description="Climate variable: 'rainfall' or 'max_temp'.")
    forecast_days: int = Field(default=7, ge=1, le=14, description="Number of days to forecast (1-14).")
    return_full_grid: bool = Field(default=False, description="If true, returns full flat lat/lon/value grid instead of summary statistics only.")
    extra_params: Optional[Dict[str, Any]] = Field(default=None, description="Reserved for future model-specific configuration parameters.")

class ConfigureResponse(BaseModel):
    status: str
    config_applied: Dict[str, Any]
    pilot_region: str
    variable: str
    forecast_days: int
    grid_resolution_degrees: float
    predicted_grid_mean: float
    predicted_grid_max: float
    predicted_grid_min: float
    predicted_grid_std: float
    units: str
    grid_points: Optional[List[Dict[str, float]]] = None

# ─────────────────────────── Endpoints ───────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "name": "India's AI Climate Digital Twin API",
        "version": "1.0.0",
        "supported_regions": list(PILOT_REGIONS.keys()),
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/api/v1/status", tags=["Info"])
def get_data_status():
    """Returns the current data assimilation status for all real sources."""
    status = {}
    for fname, label in [
        ("IMD_Gridded_Rainfall_0.25_Real_v4.nc", "IMD_Gridded_Rainfall_0.25"),
        ("IMD_Gridded_MaxTemp_1.0_Real_v3.nc", "IMD_Gridded_MaxTemp_1.0"),
        ("IMD_Gridded_MinTemp_1.0_Real_v3.nc", "IMD_Gridded_MinTemp_1.0"),
        ("MOSDAC_INSAT_LST_Real.nc", "MOSDAC_INSAT_LST"),
        ("MOSDAC_INSAT_SST_Real.nc", "MOSDAC_INSAT_SST"),
        ("MOSDAC_INSAT_Rainfall_Real.nc", "MOSDAC_INSAT_Rainfall"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        status[label] = "LOADED" if os.path.exists(path) else "NOT_AVAILABLE"
    return {"data_sources": status}

@app.get("/api/v1/forecast", response_model=ForecastResponse, tags=["Forecast"])
def get_spatial_forecast(
    variable: str = Query("rainfall", description="Climate variable: 'rainfall' or 'max_temp'"),
    days_ahead: int = Query(7, ge=1, le=14, description="Number of days to forecast (1-14)"),
    region: str = Query("Karnataka", description="Select Pilot Region")
):
    """
    Generate a short-term spatial forecast using the AI ConvLSTM engine.
    Returns summary statistics of the predicted 2D grid over the selected region.
    """
    try:
        ds_rain, ds_temp = _load_datasets(region)
        if variable == "rainfall":
            preds, _, _ = predictor.predict_rainfall_next_days_spatial(ds_rain.rainfall, days_ahead=days_ahead)
            final_grid = preds[-1]
            units = "mm/day"
            resolution = 0.25
        elif variable == "max_temp":
            preds, _, _ = predictor.predict_rainfall_next_days_spatial(ds_temp.max_temp, days_ahead=days_ahead)
            final_grid = preds[-1]
            units = "°C"
            resolution = 1.0
        else:
            raise HTTPException(status_code=400, detail="variable must be 'rainfall' or 'max_temp'")

        return ForecastResponse(
            pilot_region=region,
            variable=variable,
            days_ahead=days_ahead,
            grid_resolution_degrees=resolution,
            predicted_grid_mean=round(float(np.nanmean(final_grid)), 3),
            predicted_grid_max=round(float(np.nanmax(final_grid)), 3),
            predicted_grid_min=round(float(np.nanmin(final_grid)), 3),
            units=units
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/v1/simulate", response_model=WhatIfResponse, tags=["What-If"])
def run_what_if_simulation(
    rainfall_change_pct: float = Query(0.0, description="Percentage change in rainfall (-50 to +50)"),
    temp_change_c: float = Query(0.0, description="Temperature change in Celsius (-5.0 to +5.0)"),
    region: str = Query("Karnataka", description="Select Pilot Region")
):
    """
    Run a what-if climate scenario simulation on the selected pilot grid.
    Returns mean modified rainfall and sector-specific impact assessments for
    Agriculture (Kharif crop stress) and Reservoir Management (River basins).
    """
    try:
        ds_rain, ds_temp = _load_datasets(region)
        latest_rain = ds_rain.rainfall.isel(time=-1)
        latest_temp = ds_temp.max_temp.isel(time=-1)
        results = predictor.simulate_what_if_spatial(latest_rain, latest_temp, rainfall_change_pct, temp_change_c)

        mod_mean = float(np.nanmean(results['modified_rainfall']))
        base_mean = float(np.nanmean(latest_rain.values))
        delta = mod_mean - base_mean

        if delta < -10:
            impact = "SEVERE_DROUGHT"
            agri = f"CRITICAL: Kharif crop failure risk for rice/millets in {region}. Irrigation advisory required."
            reservoirs = f"ALERT: Major reservoir inflows in {region} critically below normal. Rationing likely."
        elif delta < -5:
            impact = "MODERATE_DRY"
            agri = f"WARNING: Below-normal rainfall stress on crops in {region}. Monitor soil moisture."
            reservoirs = f"WATCH: Reservoir storage declining in {region}. Advisory recommended."
        elif delta > 10:
            impact = "SEVERE_FLOOD_RISK"
            agri = f"ALERT: Flash flood risk to standing crops in coastal and riverine {region} districts."
            reservoirs = f"CRITICAL: Reservoir levels in {region} at overflow risk. Pre-emptive discharge protocols activate."
        elif delta > 5:
            impact = "MODERATE_WET"
            agri = "FAVORABLE: Surplus rainfall beneficial for sugarcane and paddy. Waterlogging risk in low-lying fields."
            reservoirs = "FAVORABLE: Above-normal inflows boosting storage. Track dam levels closely."
        else:
            impact = "NORMAL"
            agri = f"NORMAL: Rainfall within tolerable stress thresholds for all major crops in {region}."
            reservoirs = f"NORMAL: Reservoir storage levels in {region} trending within seasonal norms."

        return WhatIfResponse(
            pilot_region=region,
            rainfall_change_pct=rainfall_change_pct,
            temp_change_c=temp_change_c,
            modified_rainfall_mean=round(mod_mean, 3),
            impact_level=impact,
            sector_impact_agriculture=agri,
            sector_impact_reservoirs=reservoirs
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/v1/validation", response_model=ValidationResponse, tags=["AI Metrics"])
def get_validation_metrics(region: str = Query("Karnataka", description="Select Pilot Region")):
    """
    Returns RMSE and MAE of the spatial ConvLSTM model computed on the
    historical 2023 IMD Gridded Rainfall holdout dataset (last 30 days).
    """
    try:
        ds_rain, _ = _load_datasets(region)
        target_data = ds_rain.rainfall.isel(time=slice(-30, None))
        base_data = ds_rain.rainfall.isel(time=slice(0, -30))
        preds, _, _ = predictor.predict_rainfall_next_days_spatial(base_data, days_ahead=30)
        valid_mask = ~np.isnan(target_data.values)
        rmse = float(np.sqrt(np.mean((target_data.values[valid_mask] - preds[valid_mask])**2)))
        mae = float(np.mean(np.abs(target_data.values[valid_mask] - preds[valid_mask])))
        return ValidationResponse(
            pilot_region=region,
            variable="rainfall",
            dataset="IMD Gridded Rainfall 0.25 (2023 Holdout)",
            holdout_days=30,
            rmse=round(rmse, 3),
            mae=round(mae, 3),
            units="mm/day"
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/v1/alerts", response_model=AlertResponse, tags=["Alerts"])
def get_alerts(region: str = Query("Karnataka", description="Select Pilot Region")):
    """
    Returns the current automated extreme weather anomaly alerts computed from
    real IMD gridded data. Flags heatwave and flood risks using official IMD thresholds.
    """
    try:
        ds_rain, ds_temp = _load_datasets(region)
        alerts = alert_engine.compute_alerts(
            ds_rain.rainfall.isel(time=-1),
            ds_temp.max_temp.isel(time=-1)
        )
        return AlertResponse(pilot_region=region, alert_count=len(alerts), alerts=alerts)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/v1/configure", response_model=ConfigureResponse, tags=["Configuration"])
def configure_and_run_simulation(config: ConfigureRequest):
    """
    **Configure and trigger a digital twin simulation run via JSON request body.**

    This endpoint satisfies the ISRO PS-5 requirement that the digital twin provides
    "interfaces to configure the simulations, output, and data consumers".

    Submit a JSON body specifying `region`, `variable`, `forecast_days`, and whether to
    return the full spatial grid (`return_full_grid=true`). The endpoint immediately runs
    the configured forecast and returns the results alongside a config echo.

    Example request body:
    ```json
    {
      "region": "Maharashtra",
      "variable": "rainfall",
      "forecast_days": 5,
      "return_full_grid": false
    }
    ```
    """
    region = config.region
    variable = config.variable
    days = config.forecast_days

    if region not in PILOT_REGIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown region '{region}'. Valid regions: {sorted(PILOT_REGIONS.keys())}"
        )
    if variable not in ("rainfall", "max_temp"):
        raise HTTPException(
            status_code=422,
            detail="variable must be 'rainfall' or 'max_temp'"
        )

    try:
        ds_rain, ds_temp = _load_datasets(region)
        if variable == "rainfall":
            preds, _, _ = predictor.predict_rainfall_next_days_spatial(ds_rain.rainfall, days_ahead=days)
            units = "mm/day"
            resolution = 0.25
        else:
            preds, _, _ = predictor.predict_rainfall_next_days_spatial(ds_temp.max_temp, days_ahead=days)
            units = "degC"
            resolution = 1.0

        final_grid = preds[-1]   # last day forecast
        flat = final_grid[~np.isnan(final_grid)]

        grid_points = None
        if config.return_full_grid:
            if variable == "rainfall":
                lats = ds_rain.rainfall.lat.values
                lons = ds_rain.rainfall.lon.values
            else:
                lats = ds_temp.max_temp.lat.values
                lons = ds_temp.max_temp.lon.values
            import xarray as xr
            lat_g, lon_g = np.meshgrid(lats, lons, indexing='ij')
            grid_points = [
                {"lat": float(lat_g[i, j]), "lon": float(lon_g[i, j]), "value": float(final_grid[i, j])}
                for i in range(final_grid.shape[0])
                for j in range(final_grid.shape[1])
                if not np.isnan(final_grid[i, j])
            ]

        config_echo = {
            "region": region,
            "variable": variable,
            "forecast_days": days,
            "return_full_grid": config.return_full_grid,
            "extra_params": config.extra_params,
        }

        return ConfigureResponse(
            status="SUCCESS",
            config_applied=config_echo,
            pilot_region=region,
            variable=variable,
            forecast_days=days,
            grid_resolution_degrees=resolution,
            predicted_grid_mean=round(float(np.nanmean(flat)), 3),
            predicted_grid_max=round(float(np.nanmax(flat)), 3),
            predicted_grid_min=round(float(np.nanmin(flat)), 3),
            predicted_grid_std=round(float(np.nanstd(flat)), 3),
            units=units,
            grid_points=grid_points
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
