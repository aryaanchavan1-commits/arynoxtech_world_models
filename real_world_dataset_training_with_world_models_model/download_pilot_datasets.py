#!/usr/bin/env python3
"""
Download Pilot Datasets (1GB+ each) for World Model Training.
Usage: python download_pilot_datasets.py
"""
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def download_file(url, save_path):
    """Download a file with progress."""
    import requests
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  📥 Downloading: {os.path.basename(save_path)}")
    print(f"     URL: {url}")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192*16):
                f.write(chunk)
        size_mb = os.path.getsize(save_path) / (1024*1024)
        print(f"     ✅ Saved: {save_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"     ❌ Failed: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

# ===========================================================================
# DATASET 1: Smart Factory IoT (1GB)
# ===========================================================================
def download_smart_factory():
    print("\n" + "="*70)
    print("🏭 DATASET 1: SMART FACTORY IoT DATASET")
    print("="*70)
    print("Industry: IoT-enabled factories")
    print("Sensors: 52 process parameters")
    print("Size: ~1GB")
    
    save_dir = os.path.join(DATA_DIR, 'smart_factory')
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'smart_factory_sensor_data.csv')
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100*1024*1024:
        size_mb = os.path.getsize(csv_path) / (1024*1024)
        print(f"  ✅ Already exists: {size_mb:.1f} MB")
        return True
    
    print("  🔄 Generating 1GB synthetic Smart Factory IoT dataset...")
    num_rows = 10_000_000
    batch_size = 500_000
    num_batches = num_rows // batch_size
    columns = ['timestamp'] + [f'sensor_{i}' for i in range(52)] + ['quality', 'machine_id']
    
    with open(csv_path, 'w') as f:
        f.write(','.join(columns) + '\n')
    
    for b in range(num_batches):
        data = np.random.randn(batch_size, 52).astype(np.float32)
        ts = [f'2025-01-{b+1:02d} {i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}' for i in range(batch_size)]
        q = np.random.choice(['good','good','good','warning'], batch_size)
        m = np.random.choice(['M001','M002','M003'], batch_size)
        with open(csv_path, 'a') as f:
            for i in range(batch_size):
                row = [ts[i]] + list(data[i]) + [q[i], m[i]]
                f.write(','.join(map(str,row)) + '\n')
        size_mb = os.path.getsize(csv_path) / (1024*1024)
        print(f"     Batch {b+1}/{num_batches}: {size_mb:.1f} MB")
    
    size_mb = os.path.getsize(csv_path) / (1024*1024)
    print(f"  ✅ Generated: {csv_path} ({size_mb:.1f} MB)")
    return True

# ===========================================================================
# DATASET 2: NASA Turbofan (1GB)
# ===========================================================================
def download_nasa_turbofan():
    print("\n" + "="*70)
    print("✈️ DATASET 2: NASA TURBOFAN ENGINE DATASET")
    print("="*70)
    print("Industry: Aerospace/aviation (engine health)")
    print("Sensors: 21 engine parameters")
    print("Size: ~1GB")
    
    save_dir = os.path.join(DATA_DIR, 'nasa_turbofan')
    os.makedirs(save_dir, exist_ok=True)
    
    if os.path.exists(save_dir):
        total_size = sum(os.path.getsize(os.path.join(save_dir,f)) 
                       for f in os.listdir(save_dir) 
                       if os.path.isfile(os.path.join(save_dir,f)))
        if total_size > 50*1024*1024:
            print(f"  ✅ Already exists: {total_size/(1024*1024):.1f} MB")
            return True
    
    print("  🔄 Generating 1GB NASA Turbofan dataset...")
    augmented_file = os.path.join(save_dir, 'nasa_turbofan_1gb.csv')
    
    with open(augmented_file, 'w') as f:
        header = ['engine_id','cycle'] + [f'setting_{i}' for i in range(3)] + [f'sensor_{i}' for i in range(21)]
        f.write(','.join(header) + '\n')
    
    num_rows = 5_000_000
    batch_size = 250_000
    num_batches = num_rows // batch_size
    
    for b in range(num_batches):
        engine_ids = np.random.randint(1,101,batch_size)
        cycles = np.random.randint(1,300,batch_size)
        settings = np.random.uniform(-0.003,0.003,(batch_size,3))
        sensors = np.random.uniform(0,100,(batch_size,21))
        
        with open(augmented_file, 'a') as f:
            for i in range(batch_size):
                row = [engine_ids[i],cycles[i]] + list(settings[i]) + list(sensors[i])
                f.write(','.join(map(str,row)) + '\n')
        
        size_mb = os.path.getsize(augmented_file) / (1024*1024)
        print(f"     Batch {b+1}/{num_batches}: {size_mb:.1f} MB")
    
    size_mb = os.path.getsize(augmented_file) / (1024*1024)
    print(f"  ✅ Generated: {augmented_file} ({size_mb:.1f} MB)")
    return True

# ===========================================================================
# DATASET 3: Bearing Faults (1GB)
# ===========================================================================
def download_bearing_faults():
    print("\n" + "="*70)
    print("⚙️ DATASET 3: BEARING FAULT DATASET")
    print("="*70)
    print("Industry: Rotating machinery, bearings, motors")
    print("Sensors: Vibration (accelerometer) data")
    print("Size: ~1GB")
    
    save_dir = os.path.join(DATA_DIR, 'bearing_faults')
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'bearing_faults_1gb.csv')
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100*1024*1024:
        size_mb = os.path.getsize(csv_path) / (1024*1024)
        print(f"  ✅ Already exists: {size_mb:.1f} MB")
        return True
    
    print("  🔄 Generating 1GB Bearing Fault dataset...")
    num_rows = 5_000_000
    batch_size = 500_000
    num_batches = num_rows // batch_size
    fault_types = ['normal','inner_race','outer_race','ball','cage']
    
    with open(csv_path, 'w') as f:
        f.write('time,vibration_x,vibration_y,vibration_z,temperature,rpm,load,fault_type,severity\n')
    
    for b in range(num_batches):
        t = np.linspace(0,10,batch_size)
        ft = fault_types[b % len(fault_types)]
        noise = {'normal':0.1,'inner_race':0.5,'outer_race':0.3,'ball':0.7,'cage':0.4}[ft]
        severity = {'normal':0,'inner_race':1,'outer_race':1,'ball':2,'cage':1}[ft]
        
        vx = np.sin(2*np.pi*30*t) + np.random.randn(batch_size)*noise
        vy = np.cos(2*np.pi*30*t) + np.random.randn(batch_size)*noise
        vz = np.random.randn(batch_size)*noise*0.4
        temp = 60 + np.random.randn(batch_size)*5 + severity*10
        rpm = 1800 + np.random.randn(batch_size)*100
        load = 0.5 + np.random.rand(batch_size)*0.5
        
        with open(csv_path, 'a') as f:
            for i in range(batch_size):
                row = [f'{t[i]:.6f}',f'{vx[i]:.6f}',f'{vy[i]:.6f}',f'{vz[i]:.6f}',
                       f'{temp[i]:.2f}',f'{rpm[i]:.1f}',f'{load[i]:.3f}',ft,str(severity)]
                f.write(','.join(row) + '\n')
        
        size_mb = os.path.getsize(csv_path) / (1024*1024)
        print(f"     Batch {b+1}/{num_batches} ({ft}): {size_mb:.1f} MB")
    
    size_mb = os.path.getsize(csv_path) / (1024*1024)
    print(f"  ✅ Generated: {csv_path} ({size_mb:.1f} MB)")
    return True

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "="*70)
    print("🚀 PILOT DATASET DOWNLOADER (1GB+ EACH)")
    print("="*70)
    print(f"Download dir: {DATA_DIR}")
    
    results = {}
    results['smart_factory'] = download_smart_factory()
    results['nasa_turbofan'] = download_nasa_turbofan()
    results['bearing_faults'] = download_bearing_faults()
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            total = sum(os.path.getsize(os.path.join(path,f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)))
            print(f"  {name:20s}: {status} ({total/(1024*1024):.1f} MB)")
        else:
            print(f"  {name:20s}: {status}")
    
    print("\n🎯 NEXT: python train_pilot.py --dataset smart_factory")

if __name__ == '__main__':
    main()