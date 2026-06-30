"""
Offline Data Audit & Verification Utility — India's Climate Digital Twin
Verifies exactly which binary files and NetCDF datasets were successfully downloaded
before network interruption, and confirms offline presentation readiness.
"""

import os

def audit_downloaded_data():
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    
    print("====================================================================")
    print("      OFFLINE DATA AUDIT & INGESTION VERIFICATION REPORT            ")
    print("====================================================================")

    # 1. Check Raw Binary Files (imdlib downloads in rain/, tmax/, tmin/)
    print("\n--- 1. RAW BINARY FILES (data/raw/) ---")
    if os.path.exists(raw_dir):
        for sub_folder in ['rain', 'tmax', 'tmin']:
            sub_dir = os.path.join(raw_dir, sub_folder)
            if os.path.exists(sub_dir):
                grd_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.grd') or f.endswith('.bin')])
                if grd_files:
                    print(f"[OK] {sub_folder.upper()} Binary Years Found ({len(grd_files)} files):")
                    for gf in grd_files:
                        size_kb = os.path.getsize(os.path.join(sub_dir, gf)) / 1024
                        print(f"    - {gf} ({size_kb:.1f} KB)")
                else:
                    print(f"[INFO] {sub_folder.upper()} directory exists but no .grd files found inside.")
            else:
                print(f"[INFO] No `{sub_folder}` subdirectory found in data/raw/.")
                
        # Also check root of data/raw for any stray files like ind2023_rfp25.grd
        root_files = [f for f in os.listdir(raw_dir) if f.endswith('.grd')]
        if root_files:
            print(f"\n[OK] Root `data/raw/` Binary Files Found ({len(root_files)} files):")
            for rf in root_files:
                size_kb = os.path.getsize(os.path.join(raw_dir, rf)) / 1024
                print(f"    - {rf} ({size_kb:.1f} KB)")
    else:
        print("[INFO] `data/raw/` directory does not exist yet.")

    # 2. Check Processed NetCDF Files (Streamlit Dashboard & PyTorch Source)
    print("\n--- 2. PROCESSED NETCDF ARCHIVES (data/processed/) ---")
    if os.path.exists(proc_dir):
        proc_files = os.listdir(proc_dir)
        nc_files = [f for f in proc_files if f.endswith('.nc')]
        if not nc_files:
            print("[WARNING] No NetCDF files found in data/processed/.")
        else:
            print(f"Total NetCDF Archives Found: {len(nc_files)}")
            for nc in sorted(nc_files):
                size_mb = os.path.getsize(os.path.join(proc_dir, nc)) / (1024 * 1024)
                print(f"  [OK] {nc} ({size_mb:.2f} MB)")
    else:
        print("[WARNING] `data/processed/` directory does not exist.")

    # 3. Offline Demonstration Readiness Check
    print("\n--- 3. OFFLINE DEMONSTRATION READINESS ---")
    essential_files = [
        'IMD_Gridded_Rainfall_0.25_Real_v2.nc',
        'IMD_Gridded_MaxTemp_1.0_Real.nc',
        'IMD_Gridded_MinTemp_1.0_Real.nc'
    ]
    
    missing = []
    for ef in essential_files:
        if not os.path.exists(os.path.join(proc_dir, ef)):
            missing.append(ef)
            
    if not missing:
        print("[SUCCESS] All essential baseline NetCDF files are fully active on disk!")
        print("[SUCCESS] Your Digital Twin dashboard is 100% READY for offline presentation.")
    else:
        print(f"[NOTE] The following specific files were not found: {missing}")
        
    print("====================================================================")

if __name__ == "__main__":
    audit_downloaded_data()
