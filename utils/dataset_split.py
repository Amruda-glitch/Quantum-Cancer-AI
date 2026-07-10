import os
import csv
import json
import random
import sys
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def perform_dataset_split():
    """
    Reads the quality assessment report to identify corrupted or duplicate files,
    excludes them, and creates a stratified train-val-test split (70/15/15%)
    across the 9 cancer classes.
    """
    print("Performing stratified dataset split...")
    config.set_seed()
    
    # Load quality report if available, else run check
    if not os.path.exists(config.QUALITY_REPORT_JSON):
        from utils.quality_check import run_quality_assessment
        quality_report = run_quality_assessment()
    else:
        with open(config.QUALITY_REPORT_JSON, 'r') as f:
            quality_report = json.load(f)
            
    # Compile list of files to exclude (corrupted and duplicates)
    exclude_files = set(quality_report.get("corrupted_files", []))
    
    # For duplicate groups, we keep the FIRST file and exclude the rest
    for group in quality_report.get("duplicates", []):
        paths = group.get("paths", [])
        if len(paths) > 1:
            for p in paths[1:]:
                exclude_files.add(p)
                
    # Class-wise collection of clean files
    class_files = defaultdict(list)
    
    for parent_folder in os.listdir(config.DATASET_DIR):
        parent_path = os.path.join(config.DATASET_DIR, parent_folder)
        if not os.path.isdir(parent_path):
            continue
            
        for subclass in os.listdir(parent_path):
            subclass_path = os.path.join(parent_path, subclass)
            if not os.path.isdir(subclass_path):
                continue
                
            # Map subfolder to one of the 9 main classes
            class_name = config.DATASET_MAPPING.get(parent_folder, {}).get(subclass)
            if not class_name:
                continue
                
            files = [f for f in os.listdir(subclass_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Apply DEMO_MODE sampling
            if config.DEMO_MODE:
                files = files[:config.DEMO_SAMPLE_SIZE]
                
            for file in files:
                file_path = os.path.join(subclass_path, file)
                # Skip excluded files
                if file_path in exclude_files:
                    continue
                class_files[class_name].append({
                    "file_path": file_path,
                    "subclass": subclass,
                    "parent_folder": parent_folder
                })
                
    # Prepare metadata list for splitting
    metadata = []
    
    # Stratified split per main class
    for class_id, class_name in enumerate(config.CLASS_NAMES):
        items = class_files[class_name]
        random.shuffle(items)
        
        n_total = len(items)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        
        print(f"Class '{class_name}': total clean={n_total}, train={n_train}, val={n_val}, test={n_total - n_train - n_val}")
        
        for idx, item in enumerate(items):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
                
            metadata.append({
                "file_path": item["file_path"],
                "class_name": class_name,
                "class_id": class_id,
                "parent_folder": item["parent_folder"],
                "subclass": item["subclass"],
                "split": split
            })
            
    # Save split metadata to CSV
    os.makedirs(os.path.dirname(config.SPLIT_CSV), exist_ok=True)
    with open(config.SPLIT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "class_name", "class_id", "parent_folder", "subclass", "split"])
        writer.writeheader()
        writer.writerows(metadata)
        
    print(f"Stratified split saved to: {config.SPLIT_CSV}")
    print(f"Total split entries: {len(metadata)}")
    
    # Print summary of split distribution
    split_counts = defaultdict(int)
    for entry in metadata:
        split_counts[entry["split"]] += 1
    print(f"Split counts: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    
    return metadata

if __name__ == "__main__":
    perform_dataset_split()
