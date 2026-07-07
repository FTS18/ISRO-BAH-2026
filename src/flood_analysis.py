import numpy as np
import xarray as xr

def compute_flood_risk(rain_da: xr.DataArray, dem_dataset: xr.Dataset) -> xr.DataArray:
    """
    Computes flood risk zoning by fusing current rainfall intensity
    with digital elevation model (DEM) topography.
    Risk Equation:
       Risk = Rainfall (mm) * exp(-Elevation (m) / 100)
    We bound and classify the risk index:
       - 0: Low Risk (Index < 8.0)
       - 1: Moderate Risk (8.0 <= Index < 25.0)
       - 2: High Risk (25.0 <= Index < 60.0)
       - 3: Extreme Risk (Index >= 60.0)
    """
    # Align DEM coordinate grid to match rainfall data
    dem_aligned = dem_dataset.z.interp_like(rain_da, method="linear")
    
    rain_vals = rain_da.values
    elev_vals = dem_aligned.values
    
    # Fill elevation NaNs with sea level (0m) or average
    elev_vals_filled = np.nan_to_num(elev_vals, nan=0.0)
    
    # Prevent negative values (below sea level bathymetry in coastal waters) from causing extreme math
    elev_vals_bounded = np.maximum(0.0, elev_vals_filled)
    
    # Compute physical flood index
    # Areas with high rainfall and low elevation have exponentially higher risk
    flood_index = rain_vals * np.exp(-elev_vals_bounded / 120.0)
    
    # Classify categories
    risk_categories = np.zeros_like(flood_index)
    risk_categories[(flood_index >= 5.0) & (flood_index < 18.0)] = 1  # Moderate
    risk_categories[(flood_index >= 18.0) & (flood_index < 45.0)] = 2 # High
    risk_categories[flood_index >= 45.0] = 3                          # Extreme
    
    # Maintain original rainfall NaNs (like ocean mask for land-based flood risk)
    risk_categories[np.isnan(rain_vals)] = np.nan
    
    return xr.DataArray(
        risk_categories,
        coords=[rain_da.lat, rain_da.lon],
        dims=["lat", "lon"],
        name="flood_risk"
    )
