import numpy as np
import yaml
import os
import logging
from pyquaternion import Quaternion

def read_yaml_file(yaml_path, verbose=False):
    """Read YAML file and get ego pose information"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check if required keys exist
        if 'true_ego_pos' not in data:
            if verbose:
                logging.error(f"Error: YAML file missing true_ego_pos. File: {yaml_path}")
                logging.error(f"Available keys: {list(data.keys())}")
            return None
        
        # Get ego pose information
        true_ego_pos = data['true_ego_pos']
        
        # Check array length
        if len(true_ego_pos) < 6:
            if verbose:
                logging.error(f"Error: Insufficient true_ego_pos array length. File: {yaml_path}")
                logging.error(f"true_ego_pos length: {len(true_ego_pos)}")
            return None
        
        return true_ego_pos
        
    except Exception as e:
        if verbose:
            logging.error(f"Error processing YAML file {yaml_path}: {str(e)}")
        return None

def compute_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose=False):
    """Compute flow between two consecutive frames
    
    Args:
        current_yaml_path: Path to current frame YAML file
        next_yaml_path: Path to next frame YAML file
        voxel_npz_path: Path to voxel NPZ file
        verbose: Whether to show detailed information
    
    Returns:
        Flow array with shape (200, 200, 16, 3)
    """
    try:
        # Read current and next frame ego poses
        current_ego_pos = read_yaml_file(current_yaml_path, verbose)
        next_ego_pos = read_yaml_file(next_yaml_path, verbose)
        
        if current_ego_pos is None or next_ego_pos is None:
            return None
        
        # Extract position and orientation
        current_pos = current_ego_pos[:3]
        current_rot = current_ego_pos[3:]
        next_pos = next_ego_pos[:3]
        next_rot = next_ego_pos[3:]
        
        # Convert angles to radians
        current_rot = np.radians(current_rot)
        next_rot = np.radians(next_rot)
        
        # Create rotation matrices
        current_q = Quaternion(axis=[1, 0, 0], angle=current_rot[0]) * \
                   Quaternion(axis=[0, 1, 0], angle=current_rot[1]) * \
                   Quaternion(axis=[0, 0, 1], angle=current_rot[2])
        next_q = Quaternion(axis=[1, 0, 0], angle=next_rot[0]) * \
                 Quaternion(axis=[0, 1, 0], angle=next_rot[1]) * \
                 Quaternion(axis=[0, 0, 1], angle=next_rot[2])
        
        # Compute transformation matrix from current to next frame
        current_to_world = np.eye(4)
        current_to_world[:3, :3] = current_q.rotation_matrix
        current_to_world[:3, 3] = current_pos
        
        world_to_next = np.eye(4)
        world_to_next[:3, :3] = next_q.rotation_matrix
        world_to_next[:3, 3] = next_pos
        
        current_to_next = world_to_next @ np.linalg.inv(current_to_world)
        
        # Load voxel data
        voxel_data = np.load(voxel_npz_path)
        occ_label = voxel_data['occ_label']
        
        # Initialize flow array
        flow = np.zeros((200, 200, 16, 3), dtype=np.float32)
        
        # Compute flow for each voxel
        for x in range(200):
            for y in range(200):
                for z in range(16):
                    if occ_label[x, y, z] != 0:  # If voxel is occupied
                        # Convert voxel coordinates to world coordinates
                        voxel_pos = np.array([
                            (x - 100) * 0.4,  # Convert to meters
                            (y - 100) * 0.4,
                            (z - 8) * 0.4
                        ])
                        
                        # Transform point from current to next frame
                        voxel_pos_homogeneous = np.append(voxel_pos, 1)
                        next_pos_homogeneous = current_to_next @ voxel_pos_homogeneous
                        next_pos = next_pos_homogeneous[:3]
                        
                        # Convert back to voxel coordinates
                        next_x = int(next_pos[0] / 0.4 + 100)
                        next_y = int(next_pos[1] / 0.4 + 100)
                        next_z = int(next_pos[2] / 0.4 + 8)
                        
                        # Check if next position is within bounds
                        if (0 <= next_x < 200 and 
                            0 <= next_y < 200 and 
                            0 <= next_z < 16):
                            # Compute flow vector
                            flow[x, y, z] = next_pos - voxel_pos
        
        if verbose:
            logging.info(f"Successfully computed flow with shape: {flow.shape}")
        return flow
        
    except Exception as e:
        logging.error(f"Error computing flow: {str(e)}")
        return None

def compute_occ_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose=False):
    """Compute occupancy flow between two consecutive frames
    
    Args:
        current_yaml_path: Path to current frame YAML file
        next_yaml_path: Path to next frame YAML file
        voxel_npz_path: Path to voxel NPZ file
        verbose: Whether to show detailed information
    
    Returns:
        Flow array with shape (200, 200, 16, 3)
    """
    return compute_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose)

if __name__ == "__main__":
    # Set input and output paths
    current_yaml_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045\000000.yaml"
    next_yaml_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045\000001.yaml"
    voxel_npz_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045_cropped_voxel\000000_voxel.npz"
    
    # Compute flow
    flow = compute_occ_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose=True)
    
    if flow is not None:
        logging.info(f"Successfully computed flow with shape: {flow.shape}")
    else:
        logging.error("Failed to compute flow") 