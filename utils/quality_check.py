import os
import json
import hashlib
import cv2
import numpy as np
from collections import defaultdict
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_image_quality_stats(image_path):
    """
    Computes statistical properties of an image to evaluate quality and processing status.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    h, w = img.shape
    mean_val = np.mean(img)
    std_val = np.std(img)
    
    # Laplacian variance for blur detection
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Histogram entropy
    hist, _ = np.histogram(img.ravel(), 256, [0, 256])
    hist = hist / (hist.sum() + 1e-7)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return {
        "width": w,
        "height": h,
        "mean": float(mean_val),
        "std": float(std_val),
        "blur_score": float(lap_var),
        "entropy": float(entropy)
    }

def run_quality_assessment():
    """
    Scans the entire dataset for quality parameters, corrupted files, and duplicates.
    """
    print("Starting dataset quality check...")
    config.set_seed()
    
    corrupted_files = []
    duplicate_groups = defaultdict(list)
    abnormal_dim_files = []
    blurred_files = []
    
    total_scanned = 0
    subclass_stats = defaultdict(lambda: {"count": 0, "sizes": set(), "formats": set(), "errors": 0})
    
    # MD5 Hashing for duplicate detection
    md5_map = {}
    
    # Scan all directories
    for parent_folder in os.listdir(config.DATASET_DIR):
        parent_path = os.path.join(config.DATASET_DIR, parent_folder)
        if not os.path.isdir(parent_path):
            continue
            
        for subclass in os.listdir(parent_path):
            subclass_path = os.path.join(parent_path, subclass)
            if not os.path.isdir(subclass_path):
                continue
                
            files = [f for f in os.listdir(subclass_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Apply DEMO_MODE sampling
            if config.DEMO_MODE:
                files = files[:config.DEMO_SAMPLE_SIZE]
                
            for file in files:
                file_path = os.path.join(subclass_path, file)
                total_scanned += 1
                
                # Check for corruption
                try:
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                        file_md5 = hashlib.md5(file_bytes).hexdigest()
                    
                    # Duplicate check
                    if file_md5 in md5_map:
                        duplicate_groups[file_md5].append(file_path)
                        if len(duplicate_groups[file_md5]) == 1:
                            # Add the first file too
                            duplicate_groups[file_md5].insert(0, md5_map[file_md5])
                    else:
                        md5_map[file_md5] = file_path
                        
                    # Open using OpenCV to check if corrupted and get dimensions
                    img = cv2.imread(file_path)
                    if img is None:
                        corrupted_files.append(file_path)
                        subclass_stats[f"{parent_folder}/{subclass}"]["errors"] += 1
                        continue
                        
                    h, w, c = img.shape
                    ext = os.path.splitext(file)[1].upper()
                    
                    subclass_stats[f"{parent_folder}/{subclass}"]["count"] += 1
                    subclass_stats[f"{parent_folder}/{subclass}"]["sizes"].add(f"{w}x{h}")
                    subclass_stats[f"{parent_folder}/{subclass}"]["formats"].add(ext)
                    
                    # Check for abnormal dimensions (Standard is 512x512)
                    if w != 512 or h != 512:
                        abnormal_dim_files.append({
                            "path": file_path,
                            "dimensions": f"{w}x{h}"
                        })
                        
                    # Check for extreme blur (Laplacian variance < 15)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if blur_score < 15.0:
                        blurred_files.append({
                            "path": file_path,
                            "blur_score": blur_score
                        })
                        
                except Exception as e:
                    corrupted_files.append(file_path)
                    subclass_stats[f"{parent_folder}/{subclass}"]["errors"] += 1
    
    # Format duplicate results
    duplicates_list = []
    for md5, paths in duplicate_groups.items():
        duplicates_list.append({
            "md5": md5,
            "paths": paths
        })
        
    # Serialize set types for JSON
    serializable_subclass_stats = {}
    for k, v in subclass_stats.items():
        serializable_subclass_stats[k] = {
            "count": v["count"],
            "sizes": list(v["sizes"]),
            "formats": list(v["formats"]),
            "errors": v["errors"]
        }
        
    report = {
        "summary": {
            "total_images_scanned": total_scanned,
            "corrupted_images_found": len(corrupted_files),
            "duplicate_images_found": sum(len(paths) - 1 for paths in duplicate_groups.values()),
            "abnormal_dimension_images": len(abnormal_dim_files),
            "extremely_blurred_images": len(blurred_files)
        },
        "corrupted_files": corrupted_files,
        "duplicates": duplicates_list,
        "abnormal_dimensions": abnormal_dim_files,
        "blurred_files": blurred_files,
        "subclass_details": serializable_subclass_stats
    }
    
    with open(config.QUALITY_REPORT_JSON, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Dataset quality report generated: {config.QUALITY_REPORT_JSON}")
    print(f"Summary: Scanned {total_scanned} files, Corrupted: {len(corrupted_files)}, Duplicates: {report['summary']['duplicate_images_found']}, Abnormal Size: {len(abnormal_dim_files)}, Blurred: {len(blurred_files)}")
    return report

def analyze_mixed_class():
    """
    Performs specialized analysis of the Leukemia (ALL) mixed class, which contains:
    - Original images (all_benign)
    - Preprocessed images (all_pre)
    - Processed images (all_pro)
    - Augmented images (all_early / duplicate files)
    """
    print("Performing analysis on Leukemia (ALL) mixed class...")
    all_dir = os.path.join(config.DATASET_DIR, "ALL")
    if not os.path.isdir(all_dir):
        print("Leukemia (ALL) folder not found.")
        return None
        
    analysis = {
        "original": [],
        "preprocessed": [],
        "processed": [],
        "augmented": [],
        "suspicious": []
    }
    
    # Keep track of file hashes to detect augmented duplicates inside the class
    class_hashes = {}
    
    for subclass in os.listdir(all_dir):
        subclass_path = os.path.join(all_dir, subclass)
        if not os.path.isdir(subclass_path):
            continue
            
        files = [f for f in os.listdir(subclass_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if config.DEMO_MODE:
            files = files[:config.DEMO_SAMPLE_SIZE]
            
        for file in files:
            file_path = os.path.join(subclass_path, file)
            stats = get_image_quality_stats(file_path)
            if stats is None:
                analysis["suspicious"].append({
                    "path": file_path,
                    "reason": "Corrupted/Unreadable"
                })
                continue
                
            # MD5 hash for duplicate/augmentation detection
            with open(file_path, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
                
            # Logic for classification:
            # 1. Suspicious size or blur
            if stats["width"] != 512 or stats["height"] != 512:
                analysis["suspicious"].append({
                    "path": file_path,
                    "reason": f"Abnormal dimensions: {stats['width']}x{stats['height']}",
                    "stats": stats
                })
                continue
            elif stats["blur_score"] < 15.0:
                analysis["suspicious"].append({
                    "path": file_path,
                    "reason": f"Extremely blurred (score: {stats['blur_score']:.2f})",
                    "stats": stats
                })
                continue
                
            # 2. Check for duplicate hashes (augmented duplicates)
            if h in class_hashes:
                analysis["augmented"].append({
                    "path": file_path,
                    "reason": "Duplicate image content",
                    "duplicate_of": class_hashes[h],
                    "stats": stats
                })
                continue
            else:
                class_hashes[h] = file_path
                
            # 3. Categorize by subfolder and visual properties
            if subclass == "all_benign":
                analysis["original"].append({"path": file_path, "stats": stats})
            elif subclass == "all_pre":
                analysis["preprocessed"].append({"path": file_path, "stats": stats})
            elif subclass == "all_pro":
                analysis["processed"].append({"path": file_path, "stats": stats})
            elif subclass == "all_early":
                # Check if it exhibits augmentation properties (e.g. rotated/flipped boundaries)
                # Keras augmentation often leaves black borders or interpolation artifacts.
                # Let's check for black border pixels (intensity < 10) at corners
                img = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
                has_border = any(c < 10 for c in corners)
                reason = "Border/Interpolation Artifact" if has_border else "Augmented subclass sample"
                analysis["augmented"].append({"path": file_path, "reason": reason, "stats": stats})
                
    cleaning_report = {
        "summary": {
            "total_analyzed": sum(len(v) for v in analysis.values()),
            "original_count": len(analysis["original"]),
            "preprocessed_count": len(analysis["preprocessed"]),
            "processed_count": len(analysis["processed"]),
            "augmented_count": len(analysis["augmented"]),
            "suspicious_count": len(analysis["suspicious"])
        },
        "details": {
            "original": [x["path"] for x in analysis["original"][:10]],
            "preprocessed": [x["path"] for x in analysis["preprocessed"][:10]],
            "processed": [x["path"] for x in analysis["processed"][:10]],
            "augmented": [{"path": x["path"], "reason": x.get("reason", "Augmented")} for x in analysis["augmented"][:10]],
            "suspicious": [{"path": x["path"], "reason": x["reason"]} for x in analysis["suspicious"][:10]]
        }
    }
    
    with open(config.CLEANING_REPORT_JSON, 'w') as f:
        json.dump(cleaning_report, f, indent=4)
        
    print(f"Leukemia cleaning report generated: {config.CLEANING_REPORT_JSON}")
    print(f"Summary: Original: {cleaning_report['summary']['original_count']}, Preprocessed: {cleaning_report['summary']['preprocessed_count']}, Processed: {cleaning_report['summary']['processed_count']}, Augmented: {cleaning_report['summary']['augmented_count']}, Suspicious: {cleaning_report['summary']['suspicious_count']}")
    
    return cleaning_report

if __name__ == "__main__":
    run_quality_assessment()
    analyze_mixed_class()
