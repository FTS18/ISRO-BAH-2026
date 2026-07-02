import os
import glob
import h5py
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _save_nc(ds, out_path):
    """Atomically write a NetCDF4 file, removing any existing file first."""
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
    ds.to_netcdf(out_path, engine='netcdf4')
    print(f"[SUCCESS] Saved: {out_path}")


# ---------------------------------------------------------------------------
# 1. Sea Surface Temperature (3RIMG_L2B_SST)
# ---------------------------------------------------------------------------
def decode_and_regrid_sst():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir  = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)

    h5_files = glob.glob(os.path.join(raw_dir, '*L2B_SST*.h5'))
    if not h5_files:
        print("No MOSDAC SST .h5 files found in data/raw/. Skipping.")
        return

    print(f"[SST] Found {len(h5_files)} files. Regridding to 0.25 deg...")
    lons_ocean = np.arange(50.0, 100.0, 0.25)
    lats_ocean = np.arange(0.0, 30.0, 0.25)
    target_lon_grid, target_lat_grid = np.meshgrid(lons_ocean, lats_ocean)
    all_grids = []

    for fpath in h5_files:
        print(f"  Processing {os.path.basename(fpath)}...")
        try:
            with h5py.File(fpath, 'r') as f:
                sst       = f['SST'][0, :, :]
                lat_raw   = f['Latitude'][:, :]
                lon_raw   = f['Longitude'][:, :]
                lat_scale = f['Latitude'].attrs.get('scale_factor', [1.0])[0]
                lon_scale = f['Longitude'].attrs.get('scale_factor', [1.0])[0]
                lat = np.where(lat_raw == 32767, np.nan, lat_raw * lat_scale)
                lon = np.where(lon_raw == 32767, np.nan, lon_raw * lon_scale)
                valid = (sst > 200) & (sst < 350) & (lat >= 0) & (lat <= 30) & (lon >= 50) & (lon <= 100)
                if valid.sum() == 0:
                    continue
                points = np.column_stack((lon[valid], lat[valid]))
                regridded = griddata(points, sst[valid], (target_lon_grid, target_lat_grid), method='linear')
                all_grids.append(regridded)
        except Exception as exc:
            print(f"  Error: {exc}")

    if not all_grids:
        print("[SST] No valid grids extracted.")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily_mean = np.nanmean(np.array(all_grids), axis=0)

    try:
        from global_land_mask import globe
        ocean_mask = globe.is_ocean(target_lat_grid, target_lon_grid)
    except ImportError:
        ocean_mask = (target_lon_grid < 73.0) | (target_lon_grid > 84.0) | (target_lat_grid < 12.0)

    daily_mean = np.where(ocean_mask, daily_mean, np.nan)
    if np.nanmean(daily_mean) > 100:
        daily_mean -= 273.15

    times = pd.date_range(start='2024-01-01', periods=1, freq='D')
    ds = xr.Dataset(
        {'sst': (['time', 'lat', 'lon'], daily_mean[np.newaxis, :, :],
                  {'units': 'degC', 'long_name': 'INSAT-3D/3DR Sea Surface Temperature'})},
        coords={'time': times, 'lat': lats_ocean, 'lon': lons_ocean},
        attrs={'description': 'Real INSAT Sea Surface Temperature (3RIMG_L2B_SST)', 'source': 'MOSDAC INSAT-3D/3DR'}
    )
    _save_nc(ds, os.path.join(proc_dir, 'MOSDAC_INSAT_SST_Real.nc'))


# ---------------------------------------------------------------------------
# 2. Land Surface Temperature (3RIMG_L2B_LST)
# ---------------------------------------------------------------------------
def decode_and_regrid_lst():
    """
    Decode INSAT-3D/3DR Land Surface Temperature HDF5 files (product: 3RIMG_L2B_LST).
    Native resolution ~4 km. Output regridded to 0.25 deg to match IMD rainfall grids.

    HDF5 variables:
        LST         -> Land Surface Temperature (K or degC), shape (1, nlat, nlon)
        Latitude    -> Swath latitudes  (scaled short int)
        Longitude   -> Swath longitudes (scaled short int)
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir  = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)

    h5_files = glob.glob(os.path.join(raw_dir, '*L2B_LST*.h5'))
    if not h5_files:
        print("No MOSDAC LST .h5 files found in data/raw/. Skipping.")
        return

    print(f"[LST] Found {len(h5_files)} files. Regridding to 0.25 deg...")
    lons_land = np.arange(66.5, 100.0, 0.25)
    lats_land = np.arange(6.5, 38.5, 0.25)
    target_lon_grid, target_lat_grid = np.meshgrid(lons_land, lats_land)
    all_grids = []

    for fpath in h5_files:
        print(f"  Processing {os.path.basename(fpath)}...")
        try:
            with h5py.File(fpath, 'r') as f:
                lst_raw   = f['LST'][0, :, :]
                lat_raw   = f['Latitude'][:, :]
                lon_raw   = f['Longitude'][:, :]
                lat_scale = f['Latitude'].attrs.get('scale_factor', [1.0])[0]
                lon_scale = f['Longitude'].attrs.get('scale_factor', [1.0])[0]
                fill_val  = float(f['LST'].attrs.get('_FillValue', [32767])[0])

                lat = np.where(lat_raw == 32767, np.nan, lat_raw * lat_scale)
                lon = np.where(lon_raw == 32767, np.nan, lon_raw * lon_scale)
                lst = np.where(lst_raw == fill_val, np.nan, lst_raw.astype(float))

                lst_scale  = f['LST'].attrs.get('scale_factor', [1.0])[0]
                lst_offset = f['LST'].attrs.get('add_offset', [0.0])[0]
                lst = lst * lst_scale + lst_offset

                valid = (
                    np.isfinite(lst) & np.isfinite(lat) & np.isfinite(lon)
                    & (lat >= 6.5) & (lat <= 38.5)
                    & (lon >= 66.5) & (lon <= 100.0)
                    & (lst > 200) & (lst < 370)
                )
                if valid.sum() < 10:
                    print("  No valid LST points in Indian subcontinent for this file.")
                    continue

                points = np.column_stack((lon[valid], lat[valid]))
                regridded = griddata(points, lst[valid],
                                     (target_lon_grid, target_lat_grid), method='linear')
                all_grids.append(regridded)
        except Exception as exc:
            print(f"  Error: {exc}")

    if not all_grids:
        print("[LST] No valid LST grids extracted.")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily_mean = np.nanmean(np.array(all_grids), axis=0)

    if np.nanmean(daily_mean) > 100:
        daily_mean -= 273.15

    times = pd.date_range(start='2024-01-01', periods=1, freq='D')
    ds = xr.Dataset(
        {'lst': (['time', 'lat', 'lon'], daily_mean[np.newaxis, :, :],
                  {'units': 'degC', 'long_name': 'INSAT-3D/3DR Land Surface Temperature'})},
        coords={'time': times, 'lat': lats_land, 'lon': lons_land},
        attrs={'description': 'Real INSAT Land Surface Temperature (3RIMG_L2B_LST)', 'source': 'MOSDAC INSAT-3D/3DR'}
    )
    _save_nc(ds, os.path.join(proc_dir, 'MOSDAC_INSAT_LST_Real.nc'))


# ---------------------------------------------------------------------------
# 3. INSAT Merged Cloud Top Rainfall (3RIMG_L2B_IMC)
# ---------------------------------------------------------------------------
def decode_and_regrid_insat_rainfall():
    """
    Decode INSAT-3D/3DR Merged Cloud Top Rainfall HDF5 files (product: 3RIMG_L2B_IMC).
    Provides half-hourly precipitation estimates at ~4 km from cloud-top brightness
    temperature thresholding and microwave calibration.
    Output regridded to 0.25 deg and converted from mm/hr to mm/day.

    HDF5 variables:
        Rain / RAINFALL / RR  -> Precipitation (mm/hr), shape (1, nlat, nlon)
        Latitude              -> Swath latitudes
        Longitude             -> Swath longitudes
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir  = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)

    h5_files = (
        glob.glob(os.path.join(raw_dir, '*L2B_IMC*.h5')) +
        glob.glob(os.path.join(raw_dir, '*3RIMG*IMC*.h5'))
    )
    if not h5_files:
        print("No MOSDAC IMC (INSAT Rainfall) .h5 files found in data/raw/. Skipping.")
        return

    print(f"[IMC] Found {len(h5_files)} files. Regridding to 0.25 deg...")
    lons_land = np.arange(66.5, 100.0, 0.25)
    lats_land = np.arange(6.5, 38.5, 0.25)
    target_lon_grid, target_lat_grid = np.meshgrid(lons_land, lats_land)
    all_grids = []
    file_times = []

    for fpath in h5_files:
        print(f"  Processing {os.path.basename(fpath)}...")
        try:
            with h5py.File(fpath, 'r') as f:
                rain_key = None
                for candidate in ['Rain', 'RAINFALL', 'RR', 'Precipitation']:
                    if candidate in f:
                        rain_key = candidate
                        break
                if rain_key is None:
                    print(f"  No rainfall variable found. Keys: {list(f.keys())}")
                    continue

                rain_raw  = f[rain_key][0, :, :]
                lat_raw   = f['Latitude'][:, :]
                lon_raw   = f['Longitude'][:, :]
                lat_scale = f['Latitude'].attrs.get('scale_factor', [1.0])[0]
                lon_scale = f['Longitude'].attrs.get('scale_factor', [1.0])[0]
                fill_val  = float(f[rain_key].attrs.get('_FillValue', [-9999.0])[0])

                lat  = np.where(lat_raw == 32767, np.nan, lat_raw * lat_scale)
                lon  = np.where(lon_raw == 32767, np.nan, lon_raw * lon_scale)
                rain = np.where((rain_raw == fill_val) | (rain_raw < 0), np.nan, rain_raw.astype(float))
                r_scale  = f[rain_key].attrs.get('scale_factor', [1.0])[0]
                r_offset = f[rain_key].attrs.get('add_offset', [0.0])[0]
                rain = rain * r_scale + r_offset

                valid = (
                    np.isfinite(rain) & np.isfinite(lat) & np.isfinite(lon)
                    & (lat >= 6.5) & (lat <= 38.5)
                    & (lon >= 66.5) & (lon <= 100.0)
                    & (rain >= 0) & (rain < 200)
                )
                if valid.sum() < 10:
                    print("  No valid IMC points in Indian subcontinent for this file.")
                    continue

                points = np.column_stack((lon[valid], lat[valid]))
                regridded = griddata(points, rain[valid],
                                     (target_lon_grid, target_lat_grid), method='linear')
                # mm/hr -> mm/day
                all_grids.append(np.nan_to_num(regridded, nan=0.0) * 24.0)

                import re as _re
                m = _re.search(r'(\d{8})', os.path.basename(fpath))
                if m:
                    file_times.append(pd.to_datetime(m.group(1), format='%Y%m%d'))
        except Exception as exc:
            print(f"  Error: {exc}")

    if not all_grids:
        print("[IMC] No valid INSAT rainfall grids extracted.")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily_mean = np.nanmean(np.array(all_grids), axis=0)

    ref_date = file_times[0] if file_times else pd.Timestamp('2024-01-01')
    times = pd.date_range(start=ref_date, periods=1, freq='D')
    ds = xr.Dataset(
        {'rain': (['time', 'lat', 'lon'], daily_mean[np.newaxis, :, :],
                   {'units': 'mm/day', 'long_name': 'INSAT-3D/3DR Merged Cloud Top Rainfall Estimate'})},
        coords={'time': times, 'lat': lats_land, 'lon': lons_land},
        attrs={
            'description': 'Real INSAT Merged Cloud Top Rainfall (3RIMG_L2B_IMC)',
            'source': 'MOSDAC INSAT-3D/3DR',
            'processing': 'Regridded from ~4 km swath to 0.25 deg via linear interpolation. Units converted from mm/hr to mm/day.'
        }
    )
    _save_nc(ds, os.path.join(proc_dir, 'MOSDAC_INSAT_Rainfall_Real.nc'))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def decode_all():
    """Run SST, LST, and IMC decoders in sequence."""
    print("=" * 60)
    print("MOSDAC INSAT HDF5 Decoding Pipeline")
    print("=" * 60)
    decode_and_regrid_sst()
    print()
    decode_and_regrid_lst()
    print()
    decode_and_regrid_insat_rainfall()
    print()
    print("[DONE] All MOSDAC decode pipelines completed.")


if __name__ == "__main__":
    decode_all()
