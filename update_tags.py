#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update Kaggle dataset tags using Kaggle API"""

import sys
import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Force UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Initialize API
api = KaggleApi()
api.authenticate()

# Dataset 1: Japanese MLB Players Statcast
dataset1_dir = "japanese-mlb-players-statcast"
dataset1_ref = "yasunorim/japan-mlb-pitchers-batters-statcast"
dataset1_tags = ["baseball", "mlb", "sports", "sabermetrics", "statcast",
                 "pitching", "japanese-players", "japan", "analytics"]

# Dataset 2: MLB Bat Tracking
dataset2_dir = "mlb-bat-tracking"
dataset2_ref = "yasunorim/mlb-bat-tracking-2024-2025"
dataset2_tags = ["baseball", "mlb", "sports", "sabermetrics", "bat-tracking",
                 "baseball-savant", "hitting", "batters", "statcast"]

def update_dataset_tags(dataset_dir, dataset_ref, new_tags):
    """Update dataset tags by modifying metadata file and uploading"""
    metadata_file = os.path.join(dataset_dir, "dataset-metadata.json")
    abs_dir = os.path.abspath(dataset_dir)

    print(f"\nProcessing: {dataset_ref}")
    print(f"Directory: {abs_dir}")
    print(f"Target tags: {new_tags}")

    # Read current metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    current_tags = metadata.get("info", {}).get("keywords", [])
    print(f"Current tags: {current_tags}")

    # Update tags
    metadata["info"]["keywords"] = new_tags

    # Write back
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=0)

    print(f"Updated metadata file with new tags")

    # Upload via API - try with absolute path
    try:
        result = api.dataset_metadata_update(folder=abs_dir, path=metadata_file)
        print(f"[OK] Tags updated successfully! Result: {result}")
        return True
    except TypeError as e:
        # Try alternative signature
        print(f"[RETRY] Trying alternative API call format...")
        try:
            result = api.dataset_metadata_update(abs_dir)
            print(f"[OK] Tags updated successfully! Result: {result}")
            return True
        except Exception as e2:
            print(f"[ERROR] Both methods failed: {e}, {e2}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to update: {e}")
        return False

# Update both datasets
print("=" * 60)
print("Kaggle Dataset Tags Updater")
print("=" * 60)

result1 = update_dataset_tags(dataset1_dir, dataset1_ref, dataset1_tags)
result2 = update_dataset_tags(dataset2_dir, dataset2_ref, dataset2_tags)

print("\n" + "=" * 60)
print("Summary:")
print(f"Dataset 1: {'SUCCESS' if result1 else 'FAILED'}")
print(f"Dataset 2: {'SUCCESS' if result2 else 'FAILED'}")
print("=" * 60)
