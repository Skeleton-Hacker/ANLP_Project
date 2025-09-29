#!/usr/bin/env python3
"""
Utility script to manage preprocessing cache for T5 finetuning.
Usage:
    python manage_cache.py --clear          # Clear all cache
    python manage_cache.py --info           # Show cache info
    python manage_cache.py --force-reprocess # Set flag to force reprocessing
"""

import argparse
import shutil
from pathlib import Path
import os

def clear_cache(cache_dir="./cache"):
    """Clear all cached preprocessed datasets."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"✅ Cleared cache directory: {cache_path}")
    else:
        print(f"ℹ️  Cache directory doesn't exist: {cache_path}")

def show_cache_info(cache_dir="./cache"):
    """Show information about cached files."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"ℹ️  Cache directory doesn't exist: {cache_path}")
        return
    
    cache_files = list(cache_path.glob("*.pkl"))
    if not cache_files:
        print(f"ℹ️  No cache files found in {cache_path}")
        return
    
    print(f"📁 Cache directory: {cache_path}")
    total_size = 0
    for file_path in cache_files:
        size = file_path.stat().st_size
        size_mb = size / (1024 * 1024)
        total_size += size
        print(f"   - {file_path.name}: {size_mb:.1f} MB")
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"💾 Total cache size: {total_size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Manage T5 preprocessing cache")
    parser.add_argument("--clear", action="store_true", help="Clear all cache files")
    parser.add_argument("--info", action="store_true", help="Show cache information")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory path")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_cache(args.cache_dir)
    elif args.info:
        show_cache_info(args.cache_dir)
    else:
        print("Usage: python manage_cache.py --clear|--info")
        show_cache_info(args.cache_dir)

if __name__ == "__main__":
    main()