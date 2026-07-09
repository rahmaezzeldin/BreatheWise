"""
Simple script to run the complete ML pipeline.
Just run: python run_pipeline.py
"""

import sys
import os

# Import and run the main pipeline
from main_pipeline import run_full_pipeline

if __name__ == "__main__":
    print("="*60)
    print("AIR QUALITY ML PIPELINE")
    print("="*60)
    print("\nThis will:")
    print("1. Run EDA (Exploratory Data Analysis)")
    print("2. Clean the data")
    print("3. Preprocess and engineer features")
    print("4. Train 3 models (AQI, Gas, Health)")
    print("5. Evaluate models")
    print("6. Deploy models")
    print("7. Setup monitoring")
    print("\nEstimated time: 10-30 minutes")
    print("="*60)
    
    response = input("\nReady to start? (yes/no): ").lower()
    
    if response in ['yes', 'y']:
        success = run_full_pipeline(
            dataset_path="clean_air_dataset.csv",
            run_eda=True,
            run_cleaning=True
        )
        
        if success:
            print("\n" + "="*60)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nNext steps:")
            print("1. Check results in 'evaluation_outputs/' folder")
            print("2. Run dashboard: python run_dashboard.py")
        else:
            print("\n✗ Pipeline failed. Check the logs above.")
            sys.exit(1)
    else:
        print("\nCancelled.")
        sys.exit(0)
