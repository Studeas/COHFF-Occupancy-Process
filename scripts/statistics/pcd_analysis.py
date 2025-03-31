import numpy as np
import open3d as o3d
import os
import logging
from datetime import datetime
from collections import Counter

def setup_logging():
    """Setup logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'pcd_analysis_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def analyze_pcd(pcd_path):
    """
    Analyze PCD file data range, including statistics of discrete integer values in the fourth column
    """
    try:
        # Read PCD file directly
        with open(pcd_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header information
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                header_end = i + 1
                break
        
        # Read data section
        data_lines = lines[header_end:]
        points_data = []
        labels = []
        
        for line in data_lines:
            values = line.strip().split()
            if len(values) >= 4:  # Ensure at least 4 columns of data
                points_data.append([float(values[0]), float(values[1]), float(values[2])])
                labels.append(int(values[3]))  # Fourth column as discrete integer values
        
        points = np.array(points_data)
        labels = np.array(labels)
        
        # Calculate point cloud statistics
        stats = {
            'points_count': len(points),
            'x_range': [points[:, 0].min(), points[:, 0].max()],
            'y_range': [points[:, 1].min(), points[:, 1].max()],
            'z_range': [points[:, 2].min(), points[:, 2].max()],
            'label_stats': {
                'unique_values': np.unique(labels).tolist(),
                'value_counts': Counter(labels).most_common(),
                'min_value': labels.min() if len(labels) > 0 else None,
                'max_value': labels.max() if len(labels) > 0 else None
            }
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Error processing PCD file {pcd_path}: {str(e)}")
        return None

def analyze_pcd_directory(directory_path):
    """
    Analyze data range of all PCD files in the directory
    """
    all_stats = []
    
    # Traverse directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('occluded.pcd'):
                pcd_path = os.path.join(root, file)
                logging.info(f"Analyzing: {pcd_path}")
                
                stats = analyze_pcd(pcd_path)
                if stats:
                    stats['file_path'] = pcd_path
                    all_stats.append(stats)
    
    return all_stats

def save_analysis_results(stats_list, output_dir):
    """
    Save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall statistics
    summary_file = os.path.join(output_dir, 'pcd_analysis_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PCD File Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Calculate overall range for all files
        all_x_min = min(stat['x_range'][0] for stat in stats_list)
        all_x_max = max(stat['x_range'][1] for stat in stats_list)
        all_y_min = min(stat['y_range'][0] for stat in stats_list)
        all_y_max = max(stat['y_range'][1] for stat in stats_list)
        all_z_min = min(stat['z_range'][0] for stat in stats_list)
        all_z_max = max(stat['z_range'][1] for stat in stats_list)
        
        # Count all label values
        all_labels = []
        for stat in stats_list:
            all_labels.extend(stat['label_stats']['unique_values'])
        unique_labels = sorted(set(all_labels))
        
        f.write("Overall Data Range:\n")
        f.write(f"X Range: [{all_x_min:.2f}, {all_x_max:.2f}]\n")
        f.write(f"Y Range: [{all_y_min:.2f}, {all_y_max:.2f}]\n")
        f.write(f"Z Range: [{all_z_min:.2f}, {all_z_max:.2f}]\n")
        f.write(f"Total Files: {len(stats_list)}\n")
        f.write(f"Total Points: {sum(stat['points_count'] for stat in stats_list)}\n")
        f.write(f"All Label Values: {unique_labels}\n\n")
        
        # Save detailed information for each file
        f.write("Detailed Information for Each File:\n")
        f.write("-" * 50 + "\n")
        for stat in stats_list:
            f.write(f"\nFile: {stat['file_path']}\n")
            f.write(f"Point Count: {stat['points_count']}\n")
            f.write(f"X Range: [{stat['x_range'][0]:.2f}, {stat['x_range'][1]:.2f}]\n")
            f.write(f"Y Range: [{stat['y_range'][0]:.2f}, {stat['y_range'][1]:.2f}]\n")
            f.write(f"Z Range: [{stat['z_range'][0]:.2f}, {stat['z_range'][1]:.2f}]\n")
            f.write("Label Statistics:\n")
            f.write(f"  Unique Values: {stat['label_stats']['unique_values']}\n")
            f.write(f"  Min Value: {stat['label_stats']['min_value']}\n")
            f.write(f"  Max Value: {stat['label_stats']['max_value']}\n")
            f.write("  Value Distribution:\n")
            for value, count in stat['label_stats']['value_counts']:
                f.write(f"    {value}: {count} points\n")
            f.write("-" * 50 + "\n")

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting PCD file analysis")
    
    # Set input and output paths
    input_dir = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045"
    output_dir = r"C:\Users\TUF\Desktop\opv2v_process"
    
    # Analyze PCD files
    stats_list = analyze_pcd_directory(input_dir)
    
    # Save analysis results
    save_analysis_results(stats_list, output_dir)
    
    logging.info(f"Analysis complete, results saved in: {output_dir}")
    logging.info(f"Log file: {log_file}")

if __name__ == "__main__":
    main()