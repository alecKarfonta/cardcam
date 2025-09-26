#!/usr/bin/env python3
"""Quick monitoring script for fast generation."""

import time
import os
from pathlib import Path

def monitor_progress():
    output_dir = Path("data/training_50k/images")
    target = 50000
    
    print("🔍 Monitoring 50K dataset generation...")
    print(f"Target: {target:,} images")
    
    start_time = time.time()
    
    while True:
        if output_dir.exists():
            current_count = len(list(output_dir.glob("*.jpg")))
            elapsed = time.time() - start_time
            
            if current_count > 0:
                rate = current_count / elapsed
                remaining = target - current_count
                eta_minutes = (remaining / rate / 60) if rate > 0 else 0
                progress = (current_count / target) * 100
                
                print(f"\r[{time.strftime('%H:%M:%S')}] Progress: {current_count:,}/{target:,} "
                      f"({progress:.1f}%) - Rate: {rate:.1f} img/sec - ETA: {eta_minutes:.1f} min", end="")
                
                if current_count >= target:
                    print(f"\n🎉 Generation complete! {current_count:,} images in {elapsed/60:.1f} minutes")
                    break
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for generation to start...", end="")
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped")
