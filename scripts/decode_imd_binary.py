import numpy as np
import xarray as xr
import pandas as pd
import os

def decode_grd_to_nc():
    input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ind2023_rfp25.grd')
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'IMD_Gridded_Rainfall_0.25_Real.nc')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Binary file not found: {input_file}")
        
    print(f"Reading {input_file}...")
    
    # Grid specification from IMD documentation
    lon_size = 135
    lat_size = 129
    days = 365 # 2023 is not a leap year
    
    # Read the raw bytes into a numpy array (float32, little-endian)
    data = np.fromfile(input_file, dtype='<f4')
    
    # Reshape the array
    # The C code reads: for(date) for(lat) for(lon)
    # So the shape in memory is (days, lat, lon)
    try:
        data = data.reshape((days, lat_size, lon_size))
    except ValueError as e:
        raise ValueError(f"File size does not match expected dimensions. Read {data.size} floats, expected {days*lat_size*lon_size}.")
        
    # The IMD missing value is usually -999.0
    data = np.where(data == -999.0, np.nan, data)
    
    # Create coordinates
    # C code: for(j=0 ; j < 135 ; j++) lo[j] = 66.5 + j * 0.25 ;
    #         for(j=0 ; j < 129 ; j++) la[j] =  6.5 + j * 0.25 ;
    lon = 66.5 + np.arange(lon_size) * 0.25
    lat = 6.5 + np.arange(lat_size) * 0.25
    time = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            rainfall=(["time", "lat", "lon"], data)
        ),
        coords=dict(
            time=time,
            lat=(["lat"], lat),
            lon=(["lon"], lon),
        ),
        attrs=dict(
            description="Real IMD Gridded Rainfall Data (0.25 deg) for 2023",
            source="India Meteorological Department (IMD)",
            units="mm/day"
        )
    )
    
    ds.to_netcdf(output_file)
    print(f"Successfully decoded binary and saved to {output_file}")

if __name__ == "__main__":
    decode_grd_to_nc()
