import numpy as np
import logging
import os
from tqdm import tqdm

def voxelize_point_cloud(pcd_path, voxel_size=(200, 200, 16), verbose=False):
    """
    Convert PCD file to voxel grid, selecting the most frequent label value for each voxel
    
    Args:
        pcd_path: Path to PCD file
        voxel_size: Voxel grid size, default (200, 200, 16)
        verbose: Whether to show detailed information
    
    Returns:
        occ_label: Voxelized label grid with shape (200, 200, 16)
    """
    try:
        # Predefined coordinate ranges
        x_range = [-40.0, 40.0]
        y_range = [-40.0, 40.0]
        z_range = [-2.0, 4.4] 
        # Point cloud is already in ego coordinates, ground z-value is approximately less than -1
        # Voxel resolution is 0.4m, ground should be around the third voxel
        # Therefore, lower bound is set to -2, upper bound to 4.4, total length is 6.4m
        # Hence, z_range = [-2, 4.4]
        
        # Calculate voxel size for each dimension
        voxel_x = (x_range[1] - x_range[0]) / voxel_size[0]
        voxel_y = (y_range[1] - y_range[0]) / voxel_size[1]
        voxel_z = (z_range[1] - z_range[0]) / voxel_size[2]
        
        # Read PCD file
        with open(pcd_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header information
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                header_end = i + 1
                break
        
        # Use dictionary to store label counts for each voxel
        voxel_labels = {}  # Key: (x_idx, y_idx, z_idx), Value: {label: count}
        
        # Read data and perform voxelization
        for line in lines[header_end:]:
            values = line.strip().split()
            if len(values) >= 4:
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                label = int(values[3])  # Fourth column is label value
                
                # Calculate voxel indices
                x_idx = int((x - x_range[0]) / voxel_x)
                y_idx = int((y - y_range[0]) / voxel_y)
                z_idx = int((z - z_range[0]) / voxel_z)
                
                # Ensure indices are within valid range
                if (0 <= x_idx < voxel_size[0] and 
                    0 <= y_idx < voxel_size[1] and 
                    0 <= z_idx < voxel_size[2]):
                    # Update label count in voxel
                    voxel_key = (x_idx, y_idx, z_idx)
                    if voxel_key not in voxel_labels:
                        voxel_labels[voxel_key] = {}
                    if label not in voxel_labels[voxel_key]:
                        voxel_labels[voxel_key][label] = 0
                    voxel_labels[voxel_key][label] += 1
        
        # Initialize final voxel grid
        occ_label = np.zeros(voxel_size, dtype=np.uint8)
        
        # Select most frequent label for each voxel
        for (x_idx, y_idx, z_idx), label_counts in voxel_labels.items():
            if label_counts:  # If voxel contains points
                # Find label with maximum count
                max_count = max(label_counts.values())
                # Get all labels with count equal to max_count
                max_labels = [label for label, count in label_counts.items() 
                            if count == max_count]
                # If multiple maximums exist, randomly select one
                selected_label = np.random.choice(max_labels)
                occ_label[x_idx, y_idx, z_idx] = selected_label
        
        if verbose:
            logging.info(f"Voxelized occ shape: {occ_label.shape}")
        return occ_label
        
    except Exception as e:
        logging.error(f"Error processing PCD file {pcd_path}: {str(e)}")
        return None

def process_pcd_to_voxel(pcd_path, voxel_size=(200, 200, 16), verbose=False):
    """
    Process PCD file and return voxelized label grid
    
    Args:
        pcd_path: Path to PCD file
        voxel_size: Voxel grid size, default (200, 200, 16)
        verbose: Whether to show detailed information
    
    Returns:
        occ_label: Voxelized label grid with shape (200, 200, 16)
    """
    return voxelize_point_cloud(pcd_path, voxel_size, verbose)

if __name__ == "__main__":
    # Set input and output paths
    input_dir = r"C:/Users/TUF/Desktop/opv2v_process/single_data_example/2021_08_18_19_48_05/1045_cropped"
    output_dir = r"C:/Users/TUF/Desktop/opv2v_process/single_data_example/2021_08_18_19_48_05/1045_cropped_voxel"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PCD files
    pcd_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pcd')]
    
    # Process each PCD file
    for pcd_file in tqdm(pcd_files, desc="Processing files"):
        # Voxelize
        occ_label = process_pcd_to_voxel(pcd_file, verbose=True)
        
        if occ_label is not None:
            # Create data dictionary
            data_dict = {
                'occ_label': occ_label,
            }
            # Save voxelization result
            output_file = os.path.join(output_dir, os.path.basename(pcd_file).replace('.pcd', '_voxel.npz'))
            np.savez(output_file, **data_dict)
            logging.info(f"Saved: {output_file}")
        else:
            logging.error(f"Processing failed: {pcd_file}")

