import os
import json
import subprocess
import shutil
import datetime
import concurrent.futures

def download_single_dataset(dataset_id, dataset_name, creds, start_str, end_str, download_dir, mosdac_api_dir):
    # Create a unique directory for this thread to avoid config.json race conditions
    thread_dir = os.path.join(os.path.dirname(mosdac_api_dir), f"mosdac_api_{dataset_id}")
    os.makedirs(thread_dir, exist_ok=True)
    
    # Copy the execution script to the temporary directory
    shutil.copy2(os.path.join(mosdac_api_dir, "mdapi.py"), os.path.join(thread_dir, "mdapi.py"))
    
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
            "skip_user_input": True,
            "generate_error_logs": True,
            "error_logs_dir": thread_dir + "/"
        }
    }
    
    config_path = os.path.join(thread_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(api_config, f, indent=4)
        
    print(f"Starting parallel download for {dataset_name} ({dataset_id})...")
    result = subprocess.run(["python", "mdapi.py"], cwd=thread_dir, capture_output=True, text=True)
    
    # Clean up thread-specific workspace
    try:
        shutil.rmtree(thread_dir)
    except Exception:
        pass
        
    return dataset_name, result.stdout, result.stderr

def download_insat_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mosdac_api_dir = os.path.join(project_root, 'scripts', 'mosdac_api')
    creds_file = os.path.join(project_root, 'config.json')
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
    
    print(f"Syncing MOSDAC datasets from {start_str} to {end_str} in parallel...")
    
    datasets = [
        ("3RIMG_L2B_SST", "Sea Surface Temperature"),
        ("3RIMG_L2B_LST", "Land Surface Temperature"),
        ("3RIMG_L2B_IMC", "Merged Cloud Top Rainfall")
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                download_single_dataset,
                dataset_id,
                dataset_name,
                creds,
                start_str,
                end_str,
                download_dir,
                mosdac_api_dir
            )
            for dataset_id, dataset_name in datasets
        ]
        
        for future in concurrent.futures.as_completed(futures):
            dataset_name, stdout, stderr = future.result()
            print(f"\n--- Output for {dataset_name} ---")
            print(stdout)
            if stderr:
                print("--- Errors ---")
                print(stderr)
            
    print("\n--- Triggering Autonomous HDF5 Decoding Pipeline ---")
    try:
        import decode_mosdac_h5
        decode_mosdac_h5.decode_all()
    except Exception as e:
        print(f"Decoding pipeline failed: {e}")
        
if __name__ == "__main__":
    download_insat_data()
