"""
Automated Multi-Decade IMD Ingestion Microservice — India's Climate Digital Twin
Connects directly to IMD Pune binary repository via open-source `imdlib`.
Downloads decades of gridded binary data (Rainfall 0.25 deg, Max Temp 1.0 deg, Min Temp 1.0 deg)
and converts them into unified Cloud-Optimized NetCDF archives for PyTorch ConvLSTM training.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IMD_Multi_Decade_Ingestion")

def download_historical_decades(start_yr=2010, end_yr=2023):
    try:
        import imdlib as imd
    except ImportError:
        logger.error("`imdlib` is not installed. Please install it using `pip install imdlib` before running this script.")
        print("Run: .\\venv312\\Scripts\\pip.exe install imdlib")
        return

    root_dir = os.path.join(os.path.dirname(__file__), '..')
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    logger.info(f"--- Initiating Automated Multi-Decade IMD Ingestion Pipeline ({start_yr} to {end_yr}) ---")

    # 1. Download & Process Rainfall (0.25 deg)
    logger.info(f"[1/3] Downloading IMD Gridded Rainfall (0.25 deg) from {start_yr} to {end_yr}...")
    try:
        imd.get_data('rain', start_yr, end_yr, fn_format='yearwise', file_dir=raw_dir)
        rain_data = imd.open_data('rain', start_yr, end_yr, 'yearwise', raw_dir)
        ds_rain = rain_data.get_xarray()
        out_rain_path = os.path.join(proc_dir, f'IMD_Rainfall_{start_yr}_{end_yr}.nc')
        ds_rain.to_netcdf(out_rain_path)
        logger.info(f"[SUCCESS] Multi-year Rainfall NetCDF generated successfully at: {out_rain_path}")
    except Exception as e:
        logger.error(f"Failed during Rainfall ingestion: {e}")

    # 2. Download & Process Max Temperature (1.0 deg)
    logger.info(f"[2/3] Downloading IMD Gridded Max Temperature (1.0 deg) from {start_yr} to {end_yr}...")
    try:
        imd.get_data('tmax', start_yr, end_yr, fn_format='yearwise', file_dir=raw_dir)
        temp_data = imd.open_data('tmax', start_yr, end_yr, 'yearwise', raw_dir)
        ds_temp = temp_data.get_xarray()
        out_temp_path = os.path.join(proc_dir, f'IMD_MaxTemp_{start_yr}_{end_yr}.nc')
        ds_temp.to_netcdf(out_temp_path)
        logger.info(f"[SUCCESS] Multi-year Max Temp NetCDF generated successfully at: {out_temp_path}")
    except Exception as e:
        logger.error(f"Failed during Max Temp ingestion: {e}")

    # 3. Download & Process Min Temperature (1.0 deg)
    logger.info(f"[3/3] Downloading IMD Gridded Min Temperature (1.0 deg) from {start_yr} to {end_yr}...")
    try:
        imd.get_data('tmin', start_yr, end_yr, fn_format='yearwise', file_dir=raw_dir)
        mint_data = imd.open_data('tmin', start_yr, end_yr, 'yearwise', raw_dir)
        ds_mint = mint_data.get_xarray()
        out_mint_path = os.path.join(proc_dir, f'IMD_MinTemp_{start_yr}_{end_yr}.nc')
        ds_mint.to_netcdf(out_mint_path)
        logger.info(f"[SUCCESS] Multi-year Min Temp NetCDF generated successfully at: {out_mint_path}")
    except Exception as e:
        logger.error(f"Failed during Min Temp ingestion: {e}")

    logger.info("--- Automated Ingestion Pipeline Completed Successfully ---")

if __name__ == "__main__":
    # Allow command line overrides if provided (e.g. python download_multi_decade_imd.py 2000 2023)
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
    download_historical_decades(start_year, end_year)
