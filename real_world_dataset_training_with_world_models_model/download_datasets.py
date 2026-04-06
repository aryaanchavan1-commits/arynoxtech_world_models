"""
Download Datasets Script
Automatically downloads all free industrial datasets for world model training.
No registration required!
"""
import os
import requests
import sys
import zipfile
import time

def download_file(url, save_path, timeout=30):
    """Download a file with progress indicator."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        print(f"  Downloading: {os.path.basename(save_path)}")
        print(f"  URL: {url}")
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r  Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)")
                    sys.stdout.flush()
        
        print(f"\n  ✅ Saved to {save_path} ({os.path.getsize(save_path)} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n  ❌ Download failed: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def download_ai4i():
    """Download AI4I Predictive Maintenance Dataset from UCI."""
    print("\n" + "="*60)
    print("📥 Downloading AI4I Predictive Maintenance Dataset")
    print("="*60)
    
    save_dir = 'real_world_dataset_training_with_world_models_model/data/ai4i_predictive'
    save_path = os.path.join(save_dir, 'ai4i_2020.csv')
    
    # UCI direct link (verified working)
    url = "https://archive.ics.uci.edu/static/public/601/data.csv"
    
    success = download_file(url, save_path)
    
    if success:
        # Validate the download
        try:
            import pandas as pd
            df = pd.read_csv(save_path)
            print(f"\n  📊 Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
            print(f"  📋 Columns: {list(df.columns)}")
            print(f"  ✅ Dataset is valid!")
            
            # Create README
            readme_path = os.path.join(save_dir, 'README.md')
            with open(readme_path, 'w') as f:
                f.write("# AI4I 2020 Predictive Maintenance Dataset\n\n")
                f.write("## Overview\n")
                f.write("This dataset contains 10,000 instances of real-world industrial sensor data.\n\n")
                f.write("## Features\n")
                f.write("- Air temperature [K]\n")
                f.write("- Process temperature [K]\n")
                f.write("- Rotational speed [rpm]\n")
                f.write("- Torque [Nm]\n")
                f.write("- Tool wear [min]\n\n")
                f.write("## Labels\n")
                f.write("- Machine failure (0/1)\n")
                f.write("- TWF: Tool Wear Failure\n")
                f.write("- HDF: Heat Dissipation Failure\n")
                f.write("- PWF: Power Failure\n")
                f.write("- OSF: Overstrain Failure\n")
                f.write("- RNF: Random Failure\n\n")
                f.write("## Source\n")
                f.write("https://archive.ics.uci.edu/dataset/601/ai4i-2020-predictive-maintenance-dataset\n")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            return False
    
    return False

def download_nasa_turbofan():
    """Download NASA Turbofan Engine Degradation Dataset."""
    print("\n" + "="*60)
    print("📥 Downloading NASA Turbofan Dataset")
    print("="*60)
    
    save_dir = 'real_world_dataset_training_with_world_models_model/data/nasa_turbofan'
    os.makedirs(save_dir, exist_ok=True)
    
    files = {
        'train_FD001.txt': 'https://raw.githubusercontent.com/pspachos/NASA-Turbofan-Engine-Degradation-Simulation-Dataset/master/data/CMAPSSData/train_FD001.txt',
        'test_FD001.txt': 'https://raw.githubusercontent.com/pspachos/NASA-Turbofan-Engine-Degradation-Simulation-Dataset/master/data/CMAPSSData/test_FD001.txt',
        'RUL_FD001.txt': 'https://raw.githubusercontent.com/pspachos/NASA-Turbofan-Engine-Degradation-Simulation-Dataset/master/data/CMAPSSData/RUL_FD001.txt'
    }
    
    all_success = True
    for filename, url in files.items():
        save_path = os.path.join(save_dir, filename)
        if not os.path.exists(save_path):
            success = download_file(url, save_path)
            all_success = all_success and success
        else:
            print(f"  ✅ {filename} already exists")
    
    return all_success

def download_all():
    """Download all instant-access datasets."""
    print("\n" + "="*70)
    print("🚀 DATASET DOWNLOADER FOR WORLD MODEL TRAINING")
    print("="*70)
    print("\nThis will download free industrial datasets for training your world model.")
    print("No registration or login required!\n")
    
    results = {}
    
    # Download AI4I (highest priority)
    results['ai4i'] = download_ai4i()
    
    # Download NASA Turbofan
    results['nasa'] = download_nasa_turbofan()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name.upper():20s}: {status}")
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    print("\nTo train your world model on AI4I data:")
    print("  python train_ai4i.py")
    print("\nTo evaluate trained models:")
    print("  python evaluate_ai4i.py")
    print("\nTo run the full pipeline:")
    print("  python train_ai4i.py --full")

if __name__ == '__main__':
    download_all()
    