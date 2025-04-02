"""
Process point cloud
Clip points outside the range
Convert coordinates relative to lidar to coordinates relative to true_ego_pos (~-3 to ~-1)
"""


import numpy as np
import os
from tqdm import tqdm
import yaml

def read_pcd_file(pcd_path):
    """Read PCD file and return point cloud data and labels"""
    try:
        # Use numpy's loadtxt for better performance
        header = []
        with open(pcd_path, 'r') as f:
            for line in f:
                header.append(line)
                if line.strip() == 'DATA ascii':
                    break
        
        # Use numpy's loadtxt to directly read the data part
        data = np.loadtxt(pcd_path, skiprows=len(header))
        if data.shape[1] < 4:
            print(f"Error: Insufficient number of columns, need at least 4 columns, got {data.shape[1]}")
            return None, None, None
            
        points = data[:, :3]  # First three columns are xyz coordinates
        labels = data[:, 3].astype(int)  # Fourth column is labels
        
        return header, points, labels
        
    except Exception as e:
        print(f"Failed to read PCD file {pcd_path}: {str(e)}")
        return None, None, None

def write_pcd_file(pcd_path, header, points, labels):
    """Write point cloud data and labels to PCD file"""
    # Update point count information
    point_count = len(points)
    for i, line in enumerate(header):
        if line.startswith('POINTS'):
            header[i] = f"POINTS {point_count}\n"
        elif line.startswith('WIDTH'):
            header[i] = f"WIDTH {point_count}\n"
    
    with open(pcd_path, 'w') as f:
        # Write header information
        for line in header:
            f.write(line)
        # Write point cloud data
        for point, label in zip(points, labels):
            f.write(f"{point[0]} {point[1]} {point[2]} {int(label)}\n")

def get_zdiff_egolidar(yaml_path):
    """Read YAML file and get z-value difference between true_ego_pos and lidar_pose"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check if required keys exist
        if 'true_ego_pos' not in data or 'lidar_pose' not in data:
            print(f"Error: YAML file missing required keys. File: {yaml_path}")
            print(f"Available keys: {list(data.keys())}")
            return None
        
        # Get z-values from true_ego_pos and lidar_pose
        true_ego_pos = data['true_ego_pos']
        lidar_pose = data['lidar_pose']
        
        # Check array lengths
        if len(true_ego_pos) < 3 or len(lidar_pose) < 3:
            print(f"Error: Insufficient position array length. File: {yaml_path}")
            print(f"true_ego_pos length: {len(true_ego_pos)}")
            print(f"lidar_pose length: {len(lidar_pose)}")
            return None
        
        true_ego_z = true_ego_pos[2]
        lidar_z = lidar_pose[2]
        
        # Calculate z-value difference (Note: ego is below lidar, so it's negative)
        z_diff = -lidar_z + true_ego_z
        return z_diff
        
    except Exception as e:
        print(f"Error processing YAML file {yaml_path}: {str(e)}")
        return None

def crop_pcd(points, labels, x_range=(-40, 40), y_range=(-40, 40), z_diff=0):
    """裁剪点云数据并平移z轴
    
    Args:
        points: 点云坐标数组
        labels: 标签数组
        x_range: x坐标范围，默认(-40, 40)
        y_range: y坐标范围，默认(-40, 40)
        z_diff: 从lidar_pose到true_ego_pos的z值差值（正值表示向下平移）
    
    Returns:
        裁剪后的点云坐标和标签
    """
    # 创建裁剪掩码
    x_mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    y_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    mask = x_mask & y_mask
    
    # 应用掩码
    cropped_points = points[mask]
    cropped_labels = labels[mask]
    
    # 应用z轴平移（向下平移，所以减去z_diff）
    cropped_points[:, 2] -= z_diff

    return cropped_points, cropped_labels

def process_folder(input_folder, output_folder):
    """Process all PCD files in a folder
    
    Args:
        input_folder: Input folder path
        output_folder: Output folder path
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PCD files
    pcd_files = []
    yaml_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('_semantic_occluded.pcd'):
                pcd_files.append(os.path.join(root, file))
                # Get corresponding YAML file path
                yaml_file = os.path.join(root, file.replace('_semantic_occluded.pcd', '.yaml'))
                yaml_files.append(yaml_file)
    
    if not pcd_files:
        print(f"No semantic point cloud files found in {input_folder}")
        return
    
    print(f"Found {len(pcd_files)} semantic point cloud files")
    
    # Process each file
    for pcd_file, yaml_file in tqdm(zip(pcd_files, yaml_files), desc="Processing files"):
        # Read YAML file to get z-value difference
        z_diff = get_zdiff_egolidar(yaml_file)
        print(f"Z-value difference: {z_diff:.3f}")
        
        # Read PCD file
        header, points, labels = read_pcd_file(pcd_file)
        
        # Crop point cloud and translate z-axis
        cropped_points, cropped_labels = crop_pcd(points, labels, z_diff=z_diff)
        
        # Build output file path
        rel_path = os.path.relpath(pcd_file, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        
        # Create output file directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write processed file
        write_pcd_file(output_path, header, cropped_points, cropped_labels)
    
    print(f"Processing complete! Output folder: {output_folder}")

if __name__ == "__main__":
    # Set input and output folder paths
    # input_folder = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045"
    # output_folder = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045_cropped"
    

    input_folder = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045"
    output_folder = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045_cropped"
    # Process folder
    process_folder(input_folder, output_folder)