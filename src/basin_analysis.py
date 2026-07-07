import numpy as np
import pandas as pd
import xarray as xr
import json
import os
from shapely.geometry import shape, Point

def compute_basin_runoff(rain_da: xr.DataArray, geojson_path: str) -> pd.DataFrame:
    """
    Computes basin-level rainfall and reservoir inflow volume (in Million Cubic Meters, MCM)
    using the SCS Curve Number hydrological method.
    """
    if rain_da is None or not os.path.exists(geojson_path):
        return pd.DataFrame()
        
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
        
    # Get the latest daily slice of rainfall
    latest_rain = rain_da.isel(time=-1)
    
    # Generate coordinates grid
    lats = latest_rain.lat.values
    lons = latest_rain.lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    rain_vals = latest_rain.values
    flat_rain = rain_vals.flatten()
    flat_lats = lat_grid.flatten()
    flat_lons = lon_grid.flatten()
    
    # Filter out NaNs
    valid_mask = ~np.isnan(flat_rain)
    flat_rain = flat_rain[valid_mask]
    flat_lats = flat_lats[valid_mask]
    flat_lons = flat_lons[valid_mask]
    
    # Pre-build Shapely points for fast query
    points = [Point(lon, lat) for lon, lat in zip(flat_lons, flat_lats)]
    
    basin_results = []
    for feature in geojson_data['features']:
        name = feature['properties']['name']
        area = feature['properties']['area_km2']
        geom = shape(feature['geometry'])
        
        # Intersect points with basin geometry
        basin_rains = []
        for pt, r in zip(points, flat_rain):
            if geom.contains(pt):
                basin_rains.append(r)
                
        if basin_rains:
            mean_rain = float(np.mean(basin_rains))
            
            # SCS Curve Number Method for runoff depth Q (mm)
            # CN = 75 (default regional agricultural soil)
            # S = 25400 / CN - 254 = 84.67 mm
            # Ia = 0.2 * S = 16.93 mm
            # Q = (P - Ia)^2 / (P - Ia + S) if P > Ia else 0.0
            cn = 75.0
            s_val = (25400.0 / cn) - 254.0
            ia = 0.2 * s_val
            
            if mean_rain > ia:
                runoff_depth = ((mean_rain - ia) ** 2) / (mean_rain - ia + s_val)
            else:
                runoff_depth = 0.0
                
            # Volume in Million Cubic Meters (MCM)
            # 1 mm of depth over 1 km2 = 1000 m3 = 0.001 MCM
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
