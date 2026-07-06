import subprocess
import os
import sys

# Ensure relative imports inside app/ work by adding directories to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

if __name__ == "__main__":
    # Launch Streamlit on port 7860 (Hugging Face default)
    subprocess.run([
        "streamlit", "run", 
        "app/streamlit_app.py", 
        "--server.port", "7860", 
        "--server.address", "0.0.0.0"
    ])
