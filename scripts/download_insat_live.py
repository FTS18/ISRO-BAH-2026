import os
import json
import subprocess
import shutil
import datetime

def download_insat_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mosdac_api_dir = os.path.join(project_root, 'scripts', 'mosdac_api')
    creds_file = os.path.join(project_root, 'config.json')
    api_config_file = os.path.join(mosdac_api_dir, 'config.json')
    download_dir = os.path.join(project_root, 'data', 'raw')
    
    os.makedirs(download_dir, exist_ok=True)
    
    # 1. Load the user's credentials
    with open(creds_file, 'r') as f:
        creds = json.load(f)
        
    # Calculate dynamic start and end dates (last 3 days to catch live data)
    today = datetime.date.today()
    three_days_ago = today - datetime.timedelta(days=3)
    start_str = three_days_ago.strftime('%Y-%m-%d')
    end_str = today.strftime('%Y-%m-%d')
    
    print(f"Syncing MOSDAC datasets from {start_str} to {end_str}...")
    
    datasets = [
        ("3RIMG_L2B_SST", "Sea Surface Temperature"),
        ("3RIMG_L2B_LST", "Land Surface Temperature"),
        ("3RIMG_L2B_IMC", "Merged Cloud Top Rainfall")
    ]
    
    for dataset_id, dataset_name in datasets:
        print(f"\n--- Downloading {dataset_name} ({dataset_id}) ---")
        
        # 2. Configure the MOSDAC mdapi config.json
        api_config = {
            "user_credentials": {
                "username/email": creds["mosdac_username"],
                "password": creds["mosdac_password"]
            },
            "search_parameters": {
                "datasetId": dataset_id,
                "startTime": start_str,
                "endTime": end_str,
                "count": "5",
                "boundingBox": "",
                "gId": ""
            },
            "download_settings": {
                "download_path": download_dir + "/",
                "organize_by_date": False,
                "skip_user_input": True, # Very important so it doesn't hang!
                "generate_error_logs": True,
                "error_logs_dir": mosdac_api_dir + "/"
            }
        }
        
        with open(api_config_file, 'w') as f:
            json.dump(api_config, f, indent=4)
            
        print(f"Configured mdapi.py to download {dataset_id}")
        
        # 3. Run the official mdapi script
        result = subprocess.run(["python", "mdapi.py"], cwd=mosdac_api_dir, capture_output=True, text=True)
        
        print("--- Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Errors ---")
            print(result.stderr)
            
    print("\n--- Triggering Autonomous HDF5 Decoding Pipeline ---")
    try:
        import decode_mosdac_h5
        decode_mosdac_h5.decode_all()
    except Exception as e:
        print(f"Decoding pipeline failed: {e}")
        
if __name__ == "__main__":
    download_insat_data()
