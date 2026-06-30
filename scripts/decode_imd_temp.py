import numpy as np
import xarray as xr
import pandas as pd
import os

def decode_temp_grd_to_nc():
    input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'max2023.grd')
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'IMD_Gridded_MaxTemp_1.0_Real.nc')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Grid dimensions for IMD 1.0 x 1.0 degree Temperature data
    n_days = 365 # 2023 is not a leap year
    n_lat = 31
    n_lon = 31
    
    # Read binary data (Float32, Little-endian)
    data = np.fromfile(input_file, dtype='<f4')
    expected_count = n_days * n_lat * n_lon
    
    if len(data) != expected_count:
        raise ValueError(f"Expected {expected_count} values, got {len(data)}")
        
    # Reshape to (time, lat, lon)
    data = data.reshape((n_days, n_lat, n_lon))
    
    # IMD missing value for temperature is 99.9
    data[data == 99.9] = np.nan
    
    # Generate coordinates
    lats = np.linspace(7.5, 37.5, n_lat)
    lons = np.linspace(67.5, 97.5, n_lon)
    times = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars=dict(
            max_temp=(["time", "lat", "lon"], data, {"units": "degC", "long_name": "Maximum Temperature"})
        ),
        coords=dict(
            time=times,
            lat=(["lat"], lats, {"units": "degrees_north"}),
            lon=(["lon"], lons, {"units": "degrees_east"})
        ),
        attrs=dict(
            description="Real IMD Gridded Daily Maximum Temperature 1.0 x 1.0 degree",
            source="India Meteorological Department (IMD) Pune",
            year=2023
        )
    )
    
    ds.to_netcdf(output_file)
    print(f"Successfully decoded {input_file} to {output_file}")

if __name__ == "__main__":
    decode_temp_grd_to_nc()
