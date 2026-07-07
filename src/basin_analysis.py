import numpy as np
import pandas as pd
import xarray as xr
import json
import os
import warnings
from shapely.geometry import shape, Point

# Major Indian river basins: simplified bounding polygon boxes
# Real implementations use GRDC or SWAT shapefiles (we use lat/lon box approximations)
RIVER_BASINS = {
    "Ganga-Yamuna":    {"lat": (24.0, 31.5), "lon": (73.0, 88.5),  "area_km2": 861_000, "color": "#1f77b4"},
    "Brahmaputra":     {"lat": (24.0, 29.5), "lon": (89.5, 97.5),  "area_km2": 580_000, "color": "#2ca02c"},
    "Indus":           {"lat": (24.5, 36.5), "lon": (66.5, 78.0),  "area_km2": 321_000, "color": "#9467bd"},
    "Godavari":        {"lat": (16.0, 22.0), "lon": (73.5, 82.5),  "area_km2": 312_000, "color": "#8c564b"},
    "Krishna":         {"lat": (13.5, 19.5), "lon": (73.5, 81.5),  "area_km2": 258_000, "color": "#e377c2"},
    "Mahanadi":        {"lat": (19.0, 23.5), "lon": (80.5, 86.5),  "area_km2": 141_000, "color": "#7f7f7f"},
    "Narmada":         {"lat": (21.0, 24.0), "lon": (72.5, 80.5),  "area_km2":  98_000, "color": "#bcbd22"},
    "Cauvery":         {"lat": ( 9.5, 13.5), "lon": (75.0, 79.5),  "area_km2":  81_000, "color": "#17becf"},
}

def compute_basin_rainfall_accumulation(rain_da, days_back=7):
    """
    Compute total accumulated rainfall (mm) over last N days for each major basin.
    Returns list of dicts: {basin_name, total_mm, area_km2, volume_km3, color}
    """
    lats = rain_da.lat.values
    lons = rain_da.lon.values
    n = min(days_back, len(rain_da.time))
    recent = rain_da.values[-n:]   # (days, lat, lon)

    results = []
    for name, info in RIVER_BASINS.items():
        lat_min, lat_max = info["lat"]
        lon_min, lon_max = info["lon"]
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        if lat_mask.sum() == 0 or lon_mask.sum() == 0:
            continue
        basin_rain = recent[:, lat_mask, :][:, :, lon_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_mm = float(np.nansum(basin_rain) / (lat_mask.sum() * lon_mask.sum()))
        # Volume = area (km2) * depth (mm) * 1e-6 (km3)
        volume_km3 = info["area_km2"] * total_mm * 1e-6
        results.append({
            "basin": name,
            "total_mm": round(total_mm, 1),
            "area_km2": info["area_km2"],
            "volume_km3": round(volume_km3, 2),
            "color": info["color"]
        })
    results.sort(key=lambda x: -x["total_mm"])
    return results

def compute_basin_forecast_accumulation(predictions, rain_da):
    """
    Compute 7-day forecast accumulated rainfall for each major basin.
    predictions: np.ndarray (days, lat, lon)
    """
    lats = rain_da.lat.values
    lons = rain_da.lon.values
    results = []
    for name, info in RIVER_BASINS.items():
        lat_min, lat_max = info["lat"]
        lon_min, lon_max = info["lon"]
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        if lat_mask.sum() == 0 or lon_mask.sum() == 0:
            continue
        basin_pred = predictions[:, lat_mask, :][:, :, lon_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_mm = float(np.nansum(basin_pred) / (lat_mask.sum() * lon_mask.sum()))
        results.append({
            "basin": name,
            "forecast_total_mm": round(total_mm, 1),
            "area_km2": info["area_km2"],
            "color": info["color"]
        })
    results.sort(key=lambda x: -x["forecast_total_mm"])
    return results

def compute_basin_runoff(rain_da: xr.DataArray, geojson_path: str) -> pd.DataFrame:
    """
    Computes basin-level rainfall and reservoir inflow volume (in Million Cubic Meters, MCM)
    using the SCS Curve Number hydrological method.
    """
    if rain_da is None or not os.path.exists(geojson_path):
        return pd.DataFrame()
        
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
        
    latest_rain = rain_da.isel(time=-1)
    
    lats = latest_rain.lat.values
    lons = latest_rain.lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    rain_vals = latest_rain.values
    flat_rain = rain_vals.flatten()
    flat_lats = lat_grid.flatten()
    flat_lons = lon_grid.flatten()
    
    valid_mask = ~np.isnan(flat_rain)
    flat_rain = flat_rain[valid_mask]
    flat_lats = flat_lats[valid_mask]
    flat_lons = flat_lons[valid_mask]
    
    points = [Point(lon, lat) for lon, lat in zip(flat_lons, flat_lats)]
    
    basin_results = []
    for feature in geojson_data['features']:
        name = feature['properties']['name']
        area = feature['properties']['area_km2']
        geom = shape(feature['geometry'])
        
        basin_rains = []
        for pt, r in zip(points, flat_rain):
            if geom.contains(pt):
                basin_rains.append(r)
                
        if basin_rains:
            mean_rain = float(np.mean(basin_rains))
            
            cn = 75.0
            s_val = (25400.0 / cn) - 254.0
            ia = 0.2 * s_val
            
            if mean_rain > ia:
                runoff_depth = ((mean_rain - ia) ** 2) / (mean_rain - ia + s_val)
            else:
                runoff_depth = 0.0
                
            inflow_mcm = float(runoff_depth * area * 0.001)
            
            basin_results.append({
                "River Basin": name,
                "Basin Area (km²)": int(area),
                "Mean Precipitation (mm)": round(mean_rain, 2),
                "Runoff Depth (mm)": round(runoff_depth, 2),
                "Reservoir Inflow (MCM)": round(inflow_mcm, 2),
                "Major Tributaries": ", ".join(feature['properties'].get('major_tributaries', [])),
                "Outflow Mouth": feature['properties'].get('mouth', 'Bay of Bengal')
            })
            
    return pd.DataFrame(basin_results)
