import os
import shutil # For file copying
from pathlib import Path
from datetime import datetime
import json

def setup_logging(base_dir):
    """
    Setup logging files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, "merge_logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"merge_log_{timestamp}.txt")
    inconsistent_file = os.path.join(log_dir, f"inconsistent_dirs_{timestamp}.json")
    operations_file = os.path.join(log_dir, f"operations_{timestamp}.json")
    
    return log_file, inconsistent_file, operations_file

def log_message(log_file, message):
    """
    Log message to file and print to console
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

def get_directory_structure(path):
    """
    Get dataset directory structure
    Returns a dictionary where key is timestamp folder and value is list of second-level directories
    """
    structure = {}
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Ensure folder name matches timestamp format
            if len(item.split('_')) == 6:  # Check if it's a timestamp format
                sub_dirs = []
                for sub_item in os.listdir(item_path):
                    sub_item_path = os.path.join(item_path, sub_item)
                    if os.path.isdir(sub_item_path):
                        sub_dirs.append(sub_item)
                structure[item] = sorted(sub_dirs)
    return structure

def merge_datasets(source_dir1, target_dir):
    """
    Merge contents from dataset1 into dataset2
    
    Args:
        source_dir1: Path to dataset1 (smaller dataset containing yaml files)
        target_dir: Path to dataset2 (larger dataset, used as target path)
    """
    # Setup logging files
    log_file, inconsistent_file, operations_file = setup_logging(target_dir)
    
    log_message(log_file, f"Starting dataset merge")
    log_message(log_file, f"Source dataset path: {source_dir1}")
    log_message(log_file, f"Target dataset path: {target_dir}")
    
    # Record all operations
    operations = {
        "source_dir": source_dir1,
        "target_dir": target_dir,
        "copied_files": [],
        "copied_yamls": []
    }
    
    # Record inconsistent directories
    inconsistent_dirs = {
        "timestamp_inconsistencies": [],
        "subdir_inconsistencies": {}
    }
    
    log_message(log_file, "Starting dataset structure analysis...")
    
    # Get directory structure of both datasets
    structure1 = get_directory_structure(source_dir1)
    structure2 = get_directory_structure(target_dir)
    
    # Check and record inconsistent directories
    all_timestamps = set(structure1.keys()) | set(structure2.keys())
    common_timestamps = set(structure1.keys()) & set(structure2.keys())
    
    if all_timestamps - common_timestamps:
        log_message(log_file, "\nInconsistent timestamp folders:")
        for timestamp in sorted(all_timestamps - common_timestamps):
            if timestamp in structure1:
                msg = f"Only exists in dataset1: {timestamp}"
                log_message(log_file, msg)
                inconsistent_dirs["timestamp_inconsistencies"].append({
                    "timestamp": timestamp,
                    "exists_in": "dataset1"
                })
            else:
                msg = f"Only exists in dataset2: {timestamp}"
                log_message(log_file, msg)
                inconsistent_dirs["timestamp_inconsistencies"].append({
                    "timestamp": timestamp,
                    "exists_in": "dataset2"
                })
    
    log_message(log_file, "\nStarting to merge contents into dataset2...")
    merged_count = 0
    yaml_copied_count = 0
    merged_subdirs_count = 0
    skipped_subdirs_count = 0
    
    # Process all timestamp folders in dataset1
    for timestamp in sorted(structure1.keys()):
        log_message(log_file, f"\nProcessing timestamp folder: {timestamp}")
        timestamp_dir1 = os.path.join(source_dir1, timestamp)
        timestamp_dir_target = os.path.join(target_dir, timestamp)
        
        # Copy yaml files (if they exist)
        yaml_files = [f for f in os.listdir(timestamp_dir1) if f.endswith('.yaml')]
        if yaml_files:
            log_message(log_file, f"Found {len(yaml_files)} yaml files in {timestamp}")
            Path(timestamp_dir_target).mkdir(parents=True, exist_ok=True)
            for yaml_file in yaml_files:
                src_yaml = os.path.join(timestamp_dir1, yaml_file)
                dst_yaml = os.path.join(timestamp_dir_target, yaml_file)
                shutil.copy2(src_yaml, dst_yaml)
                yaml_copied_count += 1
                log_message(log_file, f"Copied yaml file: {src_yaml} -> {dst_yaml}")
                operations["copied_yamls"].append({
                    "source": src_yaml,
                    "destination": dst_yaml
                })
        
        # If timestamp exists in both datasets, process subdirectories
        if timestamp in structure2:
            subdirs1 = set(structure1[timestamp])
            subdirs2 = set(structure2[timestamp])
            
            # Find common subdirectories
            common_subdirs = subdirs1 & subdirs2
            different_subdirs = (subdirs1 | subdirs2) - common_subdirs
            
            if different_subdirs:
                log_message(log_file, f"\nDifferent subdirectories in timestamp {timestamp} (will be skipped):")
                inconsistent_dirs["subdir_inconsistencies"][timestamp] = {
                    "only_in_dataset1": list(subdirs1 - subdirs2),
                    "only_in_dataset2": list(subdirs2 - subdirs1)
                }
                
                if subdirs1 - subdirs2:
                    log_message(log_file, f"Only exists in dataset1: {sorted(subdirs1 - subdirs2)}")
                if subdirs2 - subdirs1:
                    log_message(log_file, f"Only exists in dataset2: {sorted(subdirs2 - subdirs1)}")
                skipped_subdirs_count += len(different_subdirs)
            
            # Process common subdirectories
            for subdir in common_subdirs:
                source_path1 = os.path.join(source_dir1, timestamp, subdir)
                target_path = os.path.join(timestamp_dir_target, subdir)
                log_message(log_file, f"\nProcessing subdirectory: {subdir}")
                
                # Copy files from dataset1 to dataset2
                files_copied = 0
                for file in os.listdir(source_path1):
                    src_file = os.path.join(source_path1, file)
                    if os.path.isfile(src_file):
                        dst_file = os.path.join(target_path, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            files_copied += 1
                            operations["copied_files"].append({
                                "source": src_file,
                                "destination": dst_file
                            })
                
                log_message(log_file, f"Copied {files_copied} files in subdirectory {subdir}")
                merged_subdirs_count += 1
                log_message(log_file, f"Successfully merged subdirectory: {timestamp}/{subdir}")
            
            if common_subdirs:
                merged_count += 1
                log_message(log_file, f"Completed merging timestamp {timestamp}")
    
    # Save inconsistent directory records
    with open(inconsistent_file, 'w', encoding='utf-8') as f:
        json.dump(inconsistent_dirs, f, indent=4, ensure_ascii=False)
    
    # Save operation records
    with open(operations_file, 'w', encoding='utf-8') as f:
        json.dump(operations, f, indent=4, ensure_ascii=False)
    
    # Print final statistics
    summary = f"""
Merge complete! Statistics:
Number of timestamp folders processed: {merged_count}
Number of successfully merged subdirectories: {merged_subdirs_count}
Number of skipped different subdirectories: {skipped_subdirs_count}
Number of copied yaml files: {yaml_copied_count}

Log file location: {log_file}
Inconsistent directory records: {inconsistent_file}
Operation records file: {operations_file}
"""
    log_message(log_file, summary)

if __name__ == "__main__":
    # Example usage
    source_dataset1 = r"E:\opv2v_semantic_dataset\OPV2V_unzip\train"  # Smaller dataset (containing yaml files)
    target_dataset = r"E:\semantic_data\semanticlidar_18_unzip\train"   # Larger dataset (used as target path)
    
    merge_datasets(source_dataset1, target_dataset) 