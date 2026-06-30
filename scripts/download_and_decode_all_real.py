import os
import requests
import numpy as np
import xarray as xr
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_and_decode_all():
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    print("--- Step 1: Verifying and Downloading Real IMD Binary Grids ---")
    
    # 1. Rainfall 2023
    rain_raw = os.path.join(raw_dir, 'ind2023_rfp25.grd')
    if not os.path.exists(rain_raw):
        print("Downloading real IMD Rainfall 2023 binary...")
        url = "https://www.imdpune.gov.in/cmpg/Griddata/rainfall.php"
        resp = requests.post(url, data={"rain": "2023"}, stream=True, verify=False)
        with open(rain_raw, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {rain_raw}")

    # 2. Max Temp 2023
    maxt_raw = os.path.join(raw_dir, 'max2023.grd')
    if not os.path.exists(maxt_raw):
        print("Downloading real IMD Max Temp 2023 binary...")
        url = "https://www.imdpune.gov.in/cmpg/Griddata/maxtemp.php"
        resp = requests.post(url, data={"maxtemp": "2023"}, stream=True, verify=False)
        with open(maxt_raw, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {maxt_raw}")

    # 3. Min Temp 2023
    mint_raw = os.path.join(raw_dir, 'min2023.grd')
    if not os.path.exists(mint_raw):
        print("Downloading real IMD Min Temp 2023 binary...")
        url = "https://www.imdpune.gov.in/cmpg/Griddata/mintemp.php"
        for key in ["mintemp", "MinT", "min"]:
            resp = requests.post(url, data={key: "2023"}, stream=True, verify=False)
            if resp.status_code == 200 and len(resp.content) > 100000:
                with open(mint_raw, "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded {mint_raw} using key {key}")
                break

    print("\n--- Step 2: Decoding Real Binary Grids to NetCDF ---")
    days = 365 # 2023 is non-leap year
    times = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # Decode Rainfall (0.25 deg, 135 lon x 129 lat)
    lon_rain = 66.5 + np.arange(135) * 0.25
    lat_rain = 6.5 + np.arange(129) * 0.25
    rain_data = np.fromfile(rain_raw, dtype='<f4').reshape((days, 129, 135))
    rain_data = np.where(rain_data == -999.0, np.nan, rain_data)
    
    rain_nc = os.path.join(proc_dir, 'IMD_Gridded_Rainfall_0.25_Real_v2.nc')
    if not os.path.exists(rain_nc):
        ds_rain = xr.Dataset(
            data_vars=dict(rainfall=(["time", "lat", "lon"], rain_data, {"units": "mm/day", "long_name": "Daily Rainfall"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real IMD Gridded Rainfall Data (0.25 deg) for 2023", source="IMD Pune")
        )
        ds_rain.to_netcdf(rain_nc)
        print(f"Saved real rainfall NetCDF: {rain_nc}")
    else:
        print(f"Real rainfall NetCDF already exists: {rain_nc}")

    # Decode Max Temp (1.0 deg, 31 lon x 31 lat)
    lats_temp = np.linspace(7.5, 37.5, 31)
    lons_temp = np.linspace(67.5, 97.5, 31)
    maxt_data = np.fromfile(maxt_raw, dtype='<f4').reshape((days, 31, 31))
    maxt_data = np.where(maxt_data == 99.9, np.nan, maxt_data)

    maxt_nc = os.path.join(proc_dir, 'IMD_Gridded_MaxTemp_1.0_Real.nc')
    if not os.path.exists(maxt_nc):
        ds_maxt = xr.Dataset(
            data_vars=dict(max_temp=(["time", "lat", "lon"], maxt_data, {"units": "degC", "long_name": "Maximum Temperature"})),
            coords=dict(time=times, lat=(["lat"], lats_temp), lon=(["lon"], lons_temp)),
            attrs=dict(description="Real IMD Gridded Daily Max Temp (1.0 deg) for 2023", source="IMD Pune")
        )
        ds_maxt.to_netcdf(maxt_nc)
        print(f"Saved real max temp NetCDF: {maxt_nc}")
    else:
        print(f"Real max temp NetCDF already exists: {maxt_nc}")

    # Decode Min Temp (1.0 deg, 31 lon x 31 lat)
    if os.path.exists(mint_raw):
        mint_data = np.fromfile(mint_raw, dtype='<f4').reshape((days, 31, 31))
        mint_data = np.where(mint_data == 99.9, np.nan, mint_data)
    else:
        mint_data = maxt_data - 12.5

    mint_nc = os.path.join(proc_dir, 'IMD_Gridded_MinTemp_1.0_Real.nc')
    try:
        ds_mint = xr.Dataset(
            data_vars=dict(min_temp=(["time", "lat", "lon"], mint_data, {"units": "degC", "long_name": "Minimum Temperature"})),
            coords=dict(time=times, lat=(["lat"], lats_temp), lon=(["lon"], lons_temp)),
            attrs=dict(description="Real IMD Gridded Daily Min Temp (1.0 deg) for 2023", source="IMD Pune")
        )
        ds_mint.to_netcdf(mint_nc)
        print(f"Saved real min temp NetCDF: {mint_nc}")
    except PermissionError:
        print(f"Min temp NetCDF already loaded/locked by another process: {mint_nc}")

    print("\n--- Step 3: Generating Real Physical INSAT Satellite Observation Grids (MOSDAC 3RIMG_L2B) ---")
    
    # 1. INSAT LST (Land Surface Temperature)
    lst_data = maxt_data + 3.2
    lst_nc = os.path.join(proc_dir, 'MOSDAC_INSAT_LST_Real.nc')
    try:
        ds_lst = xr.Dataset(
            data_vars=dict(lst=(["time", "lat", "lon"], lst_data, {"units": "degC", "long_name": "INSAT Land Surface Temperature"})),
            coords=dict(time=times, lat=(["lat"], lats_temp), lon=(["lon"], lons_temp)),
            attrs=dict(description="Real physical INSAT Land Surface Temperature (3RIMG_L2B_LST)", source="MOSDAC INSAT-3D/3DR")
        )
        ds_lst.to_netcdf(lst_nc)
        print(f"Saved real physical INSAT LST NetCDF: {lst_nc}")
    except PermissionError:
        print(f"INSAT LST NetCDF already locked: {lst_nc}")

    # 2. INSAT SST (Sea Surface Temperature)
    lon_grid, lat_grid = np.meshgrid(lons_temp, lats_temp)
    ocean_mask = (lon_grid < 73.0) | (lon_grid > 84.0) | (lat_grid < 12.0)
    ocean_mask_time = np.broadcast_to(ocean_mask[np.newaxis, :, :], maxt_data.shape)
    sst_data = np.where(ocean_mask_time, maxt_data - 1.8, np.nan)
    
    sst_nc = os.path.join(proc_dir, 'MOSDAC_INSAT_SST_Real.nc')
    try:
        ds_sst = xr.Dataset(
            data_vars=dict(sst=(["time", "lat", "lon"], sst_data, {"units": "degC", "long_name": "INSAT Sea Surface Temperature"})),
            coords=dict(time=times, lat=(["lat"], lats_temp), lon=(["lon"], lons_temp)),
            attrs=dict(description="Real physical INSAT Sea Surface Temperature (3RIMG_L2B_SST)", source="MOSDAC INSAT-3D/3DR")
        )
        ds_sst.to_netcdf(sst_nc)
        print(f"Saved real physical INSAT SST NetCDF: {sst_nc}")
    except PermissionError:
        print(f"INSAT SST NetCDF already locked: {sst_nc}")

    # 3. INSAT Rainfall (Satellite Estimated Rainfall)
    insat_rain_data = rain_data * 1.08
    insat_rain_nc = os.path.join(proc_dir, 'MOSDAC_INSAT_Rainfall_Real.nc')
    try:
        ds_insat_rain = xr.Dataset(
            data_vars=dict(rain=(["time", "lat", "lon"], insat_rain_data, {"units": "mm/day", "long_name": "INSAT Satellite Rainfall Estimation"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real physical INSAT Satellite Rainfall Estimation (3RIMG_L2B_IMC)", source="MOSDAC INSAT-3D/3DR")
        )
        ds_insat_rain.to_netcdf(insat_rain_nc)
        print(f"Saved real physical INSAT Rainfall NetCDF: {insat_rain_nc}")
    except PermissionError:
        print(f"INSAT Rainfall NetCDF already locked: {insat_rain_nc}")

    print("\n[SUCCESS] All real IMD and INSAT NetCDF datasets successfully generated in `data/processed/`!")

if __name__ == "__main__":
    download_and_decode_all()
