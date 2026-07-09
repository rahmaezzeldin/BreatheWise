"""
Simple script to run the dashboard.
Just run: python run_dashboard.py
"""

import sys
import os
import subprocess

# Check if models exist
if not os.path.exists("models/aqi_model.pkl"):
    print("="*60)
    print("ERROR: Models not found!")
    print("="*60)
    print("\nYou need to train the models first.")
    print("Run: python run_pipeline.py")
    print("="*60)
    sys.exit(1)

print("="*60)
print("LAUNCHING AIR QUALITY DASHBOARD")
print("="*60)
print("\nThe dashboard will open in your browser at:")
print("http://localhost:8501")
print("\nPress Ctrl+C to stop the dashboard")
print("="*60)

# Run streamlit
subprocess.run(["streamlit", "run", "dashboard.py"])
