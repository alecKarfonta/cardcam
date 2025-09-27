#!/usr/bin/env python3
"""
Monitor training data generation progress.
"""

import os
import time
import psutil
from pathlib import Path
import json
from datetime import datetime, timedelta


def get_directory_stats(directory):
    """Get statistics about a directory."""
    if not os.path.exists(directory):
        return {"files": 0, "size_mb": 0}
    
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
        for file in files:
            try:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                pass
    
    return {
        "files": total_files,
        "size_mb": total_size / (1024 * 1024)
    }


def monitor_generation(output_dir, target_train=35000, target_val=10000, check_interval=30):
    """Monitor the generation progress."""
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    
    start_time = datetime.now()
    
    print(f"🔍 Monitoring generation progress...")
    print(f"Target: {target_train:,} training + {target_val:,} validation images")
    print(f"Output directory: {output_dir}")
    print(f"Check interval: {check_interval} seconds")
    print("-" * 60)
    
    while True:
        try:
            # Get current stats
            images_stats = get_directory_stats(images_dir)
            annotations_stats = get_directory_stats(annotations_dir)
            
            # Count train and val images
            train_count = 0
            val_count = 0
            
            if images_dir.exists():
                for file in images_dir.glob("*.jpg"):
                    if file.name.startswith("train_"):
                        train_count += 1
                    elif file.name.startswith("val_"):
                        val_count += 1
            
            # Calculate progress
            total_target = target_train + target_val
            total_current = train_count + val_count
            progress_percent = (total_current / total_target * 100) if total_target > 0 else 0
            
            # Calculate ETA
            elapsed = datetime.now() - start_time
            if total_current > 0:
                rate = total_current / elapsed.total_seconds()  # images per second
                remaining = total_target - total_current
                eta_seconds = remaining / rate if rate > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
            else:
                eta = "Unknown"
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            # Print status
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{current_time}] Generation Progress:")
            print(f"  Training images: {train_count:,}/{target_train:,} ({train_count/target_train*100:.1f}%)")
            print(f"  Validation images: {val_count:,}/{target_val:,} ({val_count/target_val*100:.1f}%)")
            print(f"  Total progress: {total_current:,}/{total_target:,} ({progress_percent:.1f}%)")
            print(f"  Images size: {images_stats['size_mb']:.1f} MB")
            print(f"  Annotations: {annotations_stats['files']} files ({annotations_stats['size_mb']:.1f} MB)")
            print(f"  Elapsed time: {elapsed}")
            print(f"  ETA: {eta}")
            print(f"  System: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%, Disk {disk_usage.percent:.1f}%")
            
            # Check if complete
            if total_current >= total_target:
                print(f"\n🎉 Generation complete!")
                print(f"Final stats: {total_current:,} images in {elapsed}")
                break
            
            # Wait before next check
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            time.sleep(check_interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training data generation")
    parser.add_argument("--output_dir", type=str, default="data/large_training_dataset",
                       help="Output directory to monitor")
    parser.add_argument("--target_train", type=int, default=35000,
                       help="Target number of training images")
    parser.add_argument("--target_val", type=int, default=10000,
                       help="Target number of validation images")
    parser.add_argument("--interval", type=int, default=30,
                       help="Check interval in seconds")
    
    args = parser.parse_args()
    
    monitor_generation(
        args.output_dir,
        args.target_train,
        args.target_val,
        args.interval
    )
