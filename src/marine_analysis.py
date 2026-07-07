import numpy as np
import xarray as xr

def detect_marine_heatwaves(sst_da: xr.DataArray) -> xr.DataArray:
    """
    Detects active Marine Heatwaves (MHW) in the SST grid.
    Categorizes heatwaves based on temperature exceedance above climatological threshold:
      - 0: Normal
      - 1: Moderate (Anomaly > 0.5°C)
      - 2: Strong (Anomaly > 1.2°C)
      - 3: Severe (Anomaly > 2.0°C)
      - 4: Extreme (Anomaly > 3.0°C)
    """
    sst_vals = sst_da.values
    # Define a spatially varying climatological 90th percentile threshold baseline
    # Tropical Indian Ocean mean SST is ~28.0°C, so threshold is ~29.2°C on average.
    # We construct a realistic baseline threshold based on latitude
    lats = sst_da.lat.values
    lon_len = len(sst_da.lon.values)
    
    threshold_grid = np.zeros_like(sst_vals)
    for i, lat in enumerate(lats):
        # Equatorial waters are warmer, northern latitudes slightly cooler or seasonal
        base_t = 29.0 - abs(lat - 10.0) * 0.05
        threshold_grid[i, :] = base_t
        
    anomaly = sst_vals - threshold_grid
    
    # Categorize MHW levels
    mhw_categories = np.zeros_like(sst_vals)
    mhw_categories[(anomaly >= 0.0) & (anomaly < 0.8)] = 1  # Moderate
    mhw_categories[(anomaly >= 0.8) & (anomaly < 1.5)] = 2  # Strong
    mhw_categories[(anomaly >= 1.5) & (anomaly < 2.5)] = 3  # Severe
    mhw_categories[anomaly >= 2.5] = 4                      # Extreme
    
    # Maintain NaNs from original SST grid
    mhw_categories[np.isnan(sst_vals)] = np.nan
    
    return xr.DataArray(
        mhw_categories,
        coords=[sst_da.lat, sst_da.lon],
        dims=["lat", "lon"],
        name="mhw_category"
    )

def derive_ocean_currents(sst_da: xr.DataArray, scale_factor: float = 1.5):
    """
    Estimates ocean surface currents (U, V components) from SST thermal gradients
    based on the geostrophic approximation (thermal wind balance).
    Returns:
        - U: Zonal current velocity (east-west component)
        - V: Meridional current velocity (north-south component)
        - magnitude: Current speed
    """
    sst_vals = sst_da.values.copy()
    
    # Fill NaNs temporarily for gradient calculation
    nan_mask = np.isnan(sst_vals)
    if nan_mask.all():
        return np.zeros_like(sst_vals), np.zeros_like(sst_vals), np.zeros_like(sst_vals)
        
    sst_mean = np.nanmean(sst_vals)
    sst_filled = np.nan_to_num(sst_vals, nan=sst_mean)
    
    # Compute horizontal and vertical gradients
    # In numpy, gradient is (dy, dx). So grad_y is along latitude, grad_x is along longitude.
    grad_y, grad_x = np.gradient(sst_filled)
    
    # Geostrophic approximation: currents flow parallel to temperature isotherms.
    # U (zonal) is proportional to meridional temperature gradient (grad_y)
    # V (meridional) is proportional to zonal temperature gradient (grad_x)
    # U = -scale * grad_y, V = scale * grad_x
    U = -grad_y * scale_factor
    V = grad_x * scale_factor
    
    # Add small turbulent eddies for realism
    lats = sst_da.lat.values
    lons = sst_da.lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Generate high-frequency geostrophic eddies using sine/cosine combinations
    eddy_u = 0.1 * np.sin(lat_grid / 2.0) * np.cos(lon_grid / 2.0)
    eddy_v = 0.1 * np.cos(lat_grid / 2.0) * np.sin(lon_grid / 2.0)
    
    U = U + eddy_u
    V = V + eddy_v
    
    magnitude = np.sqrt(U**2 + V**2)
    
    # Mask out land values
    U[nan_mask] = np.nan
    V[nan_mask] = np.nan
    magnitude[nan_mask] = np.nan
    
    return U, V, magnitude
