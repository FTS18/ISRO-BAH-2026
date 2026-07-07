import numpy as np
import xarray as xr
import pandas as pd
import json
from shapely.geometry import shape, Point

def compute_cwsi(lst_da: xr.DataArray, temp_da: xr.DataArray) -> xr.DataArray:
    """
    Computes the Crop Water Stress Index (CWSI) using Land Surface Temperature (LST)
    and Air Temperature (T_air) grids.
    CWSI = (dT - dT_min) / (dT_max - dT_min)
    where:
      dT = LST - T_air
      dT_min = wet limit (crop transpiring fully)
      dT_max = dry limit (stomata fully closed)
    """
    # Align coordinates
    lst_aligned = lst_da.interp_like(temp_da, method="nearest")
    
    lst_vals = lst_aligned.values
    temp_vals = temp_da.values
    
    # dT = LST - T_air
    dT = lst_vals - temp_vals
    
    # Clean NaNs
    nan_mask = np.isnan(dT)
    dT_clean = dT[~nan_mask]
    
    if len(dT_clean) == 0:
        return xr.DataArray(np.zeros_like(dT), coords=[temp_da.lat, temp_da.lon], dims=["lat", "lon"], name="cwsi")
        
    # Standard empirical approach: use percentiles as wet/dry baseline boundaries
    dT_min = np.percentile(dT_clean, 5)
    dT_max = np.percentile(dT_clean, 95)
    
    if dT_max == dT_min:
        dT_max += 1e-5
        
    cwsi_vals = (dT - dT_min) / (dT_max - dT_min)
    cwsi_vals = np.clip(cwsi_vals, 0.0, 1.0)
    cwsi_vals[nan_mask] = np.nan
    
    return xr.DataArray(
        cwsi_vals,
        coords=[temp_da.lat, temp_da.lon],
        dims=["lat", "lon"],
        name="cwsi"
    )

def compute_eddi(temp_da: xr.DataArray, rain_da: xr.DataArray) -> xr.DataArray:
    """
    Computes the Evaporative Demand Drought Index (EDDI) proxy.
    Atmospheric demand increases with temperature and decreases with rain/humidity.
    Standardized between -3.0 (extremely wet) and +3.0 (extremely dry/thirsty atmosphere).
    """
    # Align coordinates
    rain_aligned = rain_da.interp_like(temp_da, method="nearest")
    
    t_vals = temp_da.values
    r_vals = rain_aligned.values
    
    # Estimate vapor pressure deficit and atmospheric demand proxy
    # Higher temperature = higher demand; Higher rain = lower demand/more humidity
    demand = 0.06 * t_vals - 0.015 * r_vals
    
    # Standardize demand
    nan_mask = np.isnan(demand)
    demand_clean = demand[~nan_mask]
    
    if len(demand_clean) > 0:
        mean_d = np.mean(demand_clean)
        std_d = np.std(demand_clean) if np.std(demand_clean) > 0 else 1.0
        eddi_vals = (demand - mean_d) / std_d
        eddi_vals = np.clip(eddi_vals, -3.0, 3.0)
    else:
        eddi_vals = np.zeros_like(demand)
        
    eddi_vals[nan_mask] = np.nan
    
    return xr.DataArray(
        eddi_vals,
        coords=[temp_da.lat, temp_da.lon],
        dims=["lat", "lon"],
        name="eddi"
    )

def generate_district_alerts(cwsi_da: xr.DataArray, soil_moisture_da: xr.DataArray, geojson_path: str) -> pd.DataFrame:
    """
    Parses administrative boundaries from the GeoJSON, computes mean CWSI
    and soil moisture, and returns a detailed district-level crop alert table.
    """
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except Exception:
        return pd.DataFrame() # Fallback if GeoJSON not found or invalid
        
    records = []
    
    # Prepare spatial points for vector check
    lons = cwsi_da.lon.values
    lats = cwsi_da.lat.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points_2d = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    
    cwsi_flat = cwsi_da.values.ravel()
    sm_flat = soil_moisture_da.values.ravel()
    
    # Filter out NaNs for check speed
    valid_indices = ~(np.isnan(cwsi_flat) | np.isnan(sm_flat))
    valid_points = points_2d[valid_indices]
    valid_cwsi = cwsi_flat[valid_indices]
    valid_sm = sm_flat[valid_indices]
    
    if len(valid_points) == 0:
        return pd.DataFrame()
        
    # Iterate through geojson administrative features
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        # Use NAME_1 for Indian states in the provided geojson file, and fall back to district name keys
        name = props.get("NAME_1") or props.get("district") or props.get("NAME_2") or props.get("NAME")
        if not name:
            continue
            
        geom = shape(feature.get("geometry", {}))
        
        # Identify grid points within this polygon
        inside_mask = []
        for p in valid_points:
            inside_mask.append(geom.contains(Point(p[0], p[1])))
            
        inside_mask = np.array(inside_mask)
        
        if inside_mask.sum() > 0:
            mean_cwsi = float(np.mean(valid_cwsi[inside_mask]))
            mean_sm = float(np.mean(valid_sm[inside_mask]))
            
            # Formulate Alert Levels
            if mean_cwsi > 0.75 and mean_sm < 18.0:
                alert = "Emergency"
                action = "Deploy deficit irrigation. Prioritize crop survival. Restrict non-essential water."
                color = "#EF4444" # Red
            elif mean_cwsi > 0.55 and mean_sm < 28.0:
                alert = "Warning"
                action = "Increase irrigation frequency. Apply soil mulching to mitigate water loss."
                color = "#F59E0B" # Orange
            elif mean_cwsi > 0.40 and mean_sm < 38.0:
                alert = "Watch"
                action = "Monitor reservoir levels. Optimize scheduling to run during cooler hours."
                color = "#3B82F6" # Blue
            else:
                alert = "Normal"
                action = "Regular irrigation scheduling. Crop health is optimal."
                color = "#10B981" # Green
                
            records.append({
                "Region": name,
                "Crop Stress (CWSI)": f"{mean_cwsi:.2f}",
                "Soil Moisture (%)": f"{mean_sm:.1f}%",
                "Alert Level": alert,
                "Recommended Mitigation Action": action,
                "color": color
            })
            
    # Sort regions by severity (Emergency first, Normal last)
    df = pd.DataFrame(records)
    if not df.empty:
        order = {"Emergency": 0, "Warning": 1, "Watch": 2, "Normal": 3}
        df["order"] = df["Alert Level"].map(order)
        df = df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
        
    return df
