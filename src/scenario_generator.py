import numpy as np
import xarray as xr

class ClimateScenarioInjector:
    """
    Advanced Thermodynamic Scenario Injector (What-If 2.0).
    Generates physically coupled, spatially consistent climate anomaly grids.
    """
    
    @staticmethod
    def inject_heat_dome(base_temp_grid: xr.DataArray, base_rain_grid: xr.DataArray, 
                         center_lat: float, center_lon: float, 
                         radius_deg: float = 4.0, max_anomaly: float = 5.0) -> dict:
        """
        Simulates a localized heat dome using a Gaussian spatial anomaly kernel.
        Moisture and convective precipitation are suppressed inside the dome.
        """
        temp_vals = base_temp_grid.values.copy()
        rain_vals = base_rain_grid.values.copy()
        
        # Grid lat/lon
        lats = base_temp_grid.lat.values
        lons = base_temp_grid.lon.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Calculate distance from epicenter
        dist = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
        
        # Gaussian decay kernel
        spatial_weights = np.exp(-0.5 * (dist / radius_deg)**2)
        
        # Inject temperature anomaly
        mod_temp = temp_vals + (max_anomaly * spatial_weights)
        
        # Convective suppression (precipitation decay)
        # Suppress rainfall exponentially based on heat intensity
        suppression_factor = np.exp(-0.3 * (max_anomaly * spatial_weights))
        mod_rain = rain_vals * suppression_factor
        
        return {
            'modified_rainfall': mod_rain,
            'modified_max_temp': mod_temp
        }
        
    @staticmethod
    def inject_cyclonic_vortex(base_temp_grid: xr.DataArray, base_rain_grid: xr.DataArray,
                              center_lat: float, center_lon: float,
                              radius_deg: float = 3.0, max_rain_val: float = 120.0) -> dict:
        """
        Simulates a tropical cyclone vortex center.
        Injects a high-precipitation spiral vortex band and a local temperature drop.
        """
        temp_vals = base_temp_grid.values.copy()
        rain_vals = base_rain_grid.values.copy()
        
        lats = base_temp_grid.lat.values
        lons = base_temp_grid.lon.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Distance and angle
        dist = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
        angle = np.arctan2(lat_grid - center_lat, lon_grid - center_lon)
        
        # Cyclonic spiral band equation: intensity decays with distance and spirals out
        # Max rain at core, spiral rainbands
        spiral_bands = np.sin(2.0 * np.pi * (dist / 1.5) - 3.0 * angle)
        spiral_bands = (spiral_bands + 1.0) / 2.0  # normalize 0 to 1
        
        # Rain profile: Core heavy rain + spiral rainbands decaying with distance
        core_rain = max_rain_val * np.exp(-0.5 * (dist / radius_deg)**2)
        vortex_rain = core_rain * (0.6 + 0.4 * spiral_bands)
        
        # Combine with baseline observed rainfall
        mod_rain = np.maximum(rain_vals, vortex_rain)
        
        # Temperature drop due to cloud shielding and convective cooling
        temp_drop = -4.0 * np.exp(-0.5 * (dist / radius_deg)**2)
        mod_temp = temp_vals + temp_drop
        
        return {
            'modified_rainfall': mod_rain,
            'modified_max_temp': mod_temp
        }
