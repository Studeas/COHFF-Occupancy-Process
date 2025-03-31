import os
import yaml
import numpy as np
import pickle
import logging
from datetime import datetime
from pathlib import Path
import shutil
import math
from scripts.operation.clip_transform import crop_pcd, read_pcd_file, read_yaml_file
from scripts.operation.compute_occ_flow import compute_occ_flow
from scripts.operation.voxelization import voxelize_point_cloud as voxelization
import argparse
from pyquaternion import Quaternion
from tqdm import tqdm

# Constants
FREE_LABEL = 10  # Free space label

# Label mapping dictionary
OPV2V_TO_OURS_LABEL_MAPPING = {
    0: 10,  # unlabeled -> FREE
    1: 9,   # Building -> BUILDING
    2: 0,   # Fence -> GENERAL_OBJECT
    3: 10,  # unlabeled -> FREE
    4: 10,  # unlabeled -> FREE
    5: 0,   # Pole -> GENERAL_OBJECT
    6: 7,   # Road -> ROAD
    7: 7,   # Road -> ROAD
    8: 8,   # SideWalk -> WALKABLE/TERRAIN
    9: 6,   # Vegetation -> VEGETATION
    10: 1,  # Vehicles -> VEHICLE
    11: 9,  # Wall -> BUILDING
    12: 0,  # TrafficSign -> GENERAL_OBJECT
    13: 10, # unlabeled -> FREE
    14: 8,  # Ground -> WALKABLE/TERRAIN
    15: 0,  # Bridge -> GENERAL_OBJECT
    16: 10, # unlabeled -> FREE
    17: 0,  # GuardRail -> GENERAL_OBJECT
    18: 0,  # Pole -> GENERAL_OBJECT
    19: 10, # unlabeled -> FREE
    20: 1,  # Vehicles -> VEHICLE
    21: 10, # unlabeled -> FREE
    22: 8   # Terrain -> WALKABLE/TERRAIN
}

def setup_logging(output_root, verbose=False):
    """Setup logging configuration"""
    log_dir = os.path.join(output_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'processing_{timestamp}.log')
    
    # Set logging level based on verbose parameter
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_yaml(file_path, verbose=False):
    """Load YAML file"""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        if verbose:
            logging.info(f"Successfully loaded YAML file: {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load YAML file {file_path}: {str(e)}")
        return None

def remap_labels(occ_label):
    """Remap labels according to the mapping dictionary"""
    remapped_label = np.zeros_like(occ_label)
    for old_label, new_label in OPV2V_TO_OURS_LABEL_MAPPING.items():
        remapped_label[occ_label == old_label] = new_label
    return remapped_label

def construct_surround_sensor_fov_mask(occ_label, free_label, sensor_voxel_coord, v_fov):
    """
    Constructs a FOV mask for a surround-view sensor in a voxel grid.

    Parameters:
        occ_label (np.ndarray): (L, W, H) occupancy grid.
        free_label (int or float): Value representing free space.
        sensor_voxel_coord (tuple): (x, y, z) sensor position in voxel space.
        v_fov (list or tuple of two floats): [min_elevation_deg, max_elevation_deg]

    Returns:
        np.ndarray: Boolean (L, W, H) mask where True = visible (within FOV).
    """
    L, W, H = occ_label.shape
    fov_mask = np.zeros_like(occ_label, dtype=bool)
    origin = np.array(sensor_voxel_coord).astype(np.float32)

    # Convert vertical FOV range from degrees to radians
    v_min_rad = np.radians(v_fov[0])
    v_max_rad = np.radians(v_fov[1])

    # Ray sampling resolution (adjust as needed)
    num_azimuth = 720  # horizontal resolution
    num_elevation = v_fov[1] - v_fov[0]  # vertical resolution

    directions = []
    for theta in np.linspace(0, 2 * np.pi, num_azimuth, endpoint=False):  # azimuth (yaw)
        for phi in np.linspace(v_min_rad, v_max_rad, num_elevation):  # elevation (pitch)
            dx = np.cos(phi) * np.cos(theta)
            dy = np.cos(phi) * np.sin(theta)
            dz = np.sin(phi)
            directions.append(np.array([dx, dy, dz]))

    directions = np.array(directions)

    max_distance = np.linalg.norm([L, W, H])

    for dir_vec in directions:
        pos = origin.copy()
        for _ in range(int(max_distance * 2)):
            pos += dir_vec * 0.5  # step size (smaller = more accurate)

            ix, iy, iz = np.round(pos).astype(int)

            if not (0 <= ix < L and 0 <= iy < W and 0 <= iz < H):
                break  # Exits the grid

            fov_mask[ix, iy, iz] = True

            if occ_label[ix, iy, iz] != free_label:
                break  # Hit something

    return fov_mask

def process_frame(frame_yaml_path, frame_pcd_path, next_yaml_path=None, next_pcd_path=None, verbose=False):
    """Process single frame data"""
    if verbose:
        logging.info(f"Processing frame: {os.path.basename(frame_yaml_path)}")
    
    # Check if files exist
    if not os.path.exists(frame_yaml_path):
        logging.error(f"YAML file does not exist: {frame_yaml_path}")
        return None
    if not os.path.exists(frame_pcd_path):
        logging.error(f"PCD file does not exist: {frame_pcd_path}")
        return None
    
    # Load YAML data
    frame_data = load_yaml(frame_yaml_path, verbose)
    if frame_data is None:
        return None
    
    try:
        # 1. Crop and transform point cloud
        # Read YAML file to get z-value difference
        z_diff = read_yaml_file(frame_yaml_path)
        if z_diff is None:
            logging.error(f"Failed to get z-value difference: {frame_yaml_path}")
            return None
            
        # Read PCD file
        header, points, labels = read_pcd_file(frame_pcd_path)
        if header is None or points is None or labels is None:
            logging.error(f"Failed to read PCD file: {frame_pcd_path}")
            return None
            
        # Crop point cloud and translate z-axis
        cropped_points, cropped_labels = crop_pcd(points, labels, z_diff=z_diff)
        
        if cropped_points is None or len(cropped_points) == 0:
            logging.error(f"Failed to crop and transform point cloud: {frame_pcd_path}")
            return None
            
        # 2. Voxelization
        occ_label = voxelization(frame_pcd_path)  # Use original PCD file path
        if occ_label is None:
            logging.error(f"Failed to voxelize: {frame_pcd_path}")
            return None
            
        # 2.1 Remap labels
        occ_label = remap_labels(occ_label)
            
        # 3. Compute occ_flow (if next frame exists)
        occ_flow = np.zeros((200, 200, 16, 3), dtype=np.float32)
        if next_yaml_path and next_pcd_path and os.path.exists(next_yaml_path) and os.path.exists(next_pcd_path):
            try:
                # Validate voxel label format
                if occ_label.shape != (200, 200, 16):
                    logging.error(f"Invalid voxel label shape: {occ_label.shape}, expected (200, 200, 16)")
                    return None
                    
                # Validate label value range
                unique_labels = np.unique(occ_label)
                if verbose:
                    logging.info(f"Unique values in voxel labels: {unique_labels}")
                
                # Create temporary file to save voxel labels
                temp_npz_path = os.path.join(os.path.dirname(frame_pcd_path), "temp_voxel.npz")
                np.savez(temp_npz_path, occ_label=occ_label)
                
                # Validate pose information in YAML file
                next_frame_data = load_yaml(next_yaml_path, verbose)
                if next_frame_data is None:
                    logging.error(f"Failed to load next frame YAML file: {next_yaml_path}")
                    return None
                    
                if 'true_ego_pos' not in next_frame_data:
                    logging.error(f"Next frame YAML file missing true_ego_pos: {next_yaml_path}")
                    return None
                    
                # Compute flow
                occ_flow = compute_occ_flow(frame_yaml_path, next_yaml_path, temp_npz_path)
                
                # Validate flow shape and value range
                if occ_flow.shape != (200, 200, 16, 3):
                    logging.error(f"Invalid flow shape: {occ_flow.shape}, expected (200, 200, 16, 3)")
                    return None
                    
                if verbose:
                    logging.info(f"Flow value range: min={np.min(occ_flow)}, max={np.max(occ_flow)}")
                
                # Delete temporary file
                if os.path.exists(temp_npz_path):
                    os.remove(temp_npz_path)
                if verbose:
                    logging.info("Successfully computed occ_flow")
            except Exception as e:
                logging.error(f"Failed to compute occ_flow: {str(e)}")
                logging.error(f"Error details: {e.__class__.__name__}")
                import traceback
                logging.error(f"Stack trace:\n{traceback.format_exc()}")
                return None
        else:
            if verbose:
                logging.info("No next frame data, setting flow to 0")
            
        # 4. Extract ego_to_world_transformation_matrix from yaml
        true_ego_pos = frame_data.get('true_ego_pos')
        if true_ego_pos is None:
            logging.error(f"YAML file missing true_ego_pos: {frame_yaml_path}")
            return None
            
        # Check true_ego_pos format
        if not isinstance(true_ego_pos, (list, tuple, np.ndarray)):
            logging.error(f"Invalid true_ego_pos format, expected list or array: {type(true_ego_pos)}")
            return None
            
        if len(true_ego_pos) < 6:  # Check for complete 6 values
            logging.error(f"Invalid true_ego_pos length, expected 6 values (position and orientation), got {len(true_ego_pos)}")
            logging.error(f"true_ego_pos content: {true_ego_pos}")
            return None
            
        try:
            # Get position and orientation
            x, y, z = true_ego_pos[:3]
            roll, pitch, yaw = true_ego_pos[3:]
            if verbose:
                logging.info(f"Successfully got ego pose: position=[{x}, {y}, {z}], orientation=[{roll}, {pitch}, {yaw}]")
            
            # Build complete transformation matrix
            ego_to_world_transformation_matrix = np.eye(4, dtype=np.float32)
            # Set translation
            ego_to_world_transformation_matrix[:3, 3] = [x, y, z]
            # Set rotation (using Euler angles)
            # Convert angles to radians
            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            # Create rotation matrix
            q = Quaternion(axis=[1, 0, 0], angle=roll_rad) * \
                Quaternion(axis=[0, 1, 0], angle=pitch_rad) * \
                Quaternion(axis=[0, 0, 1], angle=yaw_rad)
            ego_to_world_transformation_matrix[:3, :3] = q.rotation_matrix
        except Exception as e:
            logging.error(f"Failed to process true_ego_pos: {str(e)}")
            logging.error(f"true_ego_pos content: {true_ego_pos}")
            return None
            
        # 5. Process annotation information
        annotations = []
        if "vehicles" in frame_data:
            for token, veh in frame_data["vehicles"].items():
                try:
                    location = veh.get('location', [0, 0, 0])
                    angle = veh.get('angle', [0, 0, 0])
                    
                    agent_to_ego = np.eye(4, dtype=np.float32)
                    agent_to_ego[:3, 3] = [
                        location[0] - x,
                        location[1] - y,
                        location[2] - z
                    ]
                    
                    ann = {
                        "token": token,
                        "agent_to_ego": agent_to_ego,
                        "size": np.array(veh.get("extent", [0, 0, 0]), dtype=np.float32),
                        "category": "vehicle"
                    }
                    annotations.append(ann)
                except Exception as e:
                    logging.warning(f"Error processing vehicle {token} annotation: {str(e)}")
                    continue
        
        # 6. Generate FOV mask
        # 6.1 LiDAR FOV mask
        lidar_voxel_coord = (100, 100, 10)  # LiDAR position in voxel space
        occ_mask_lidar = construct_surround_sensor_fov_mask(
            occ_label, 
            FREE_LABEL, 
            lidar_voxel_coord,
            [-30, 10]  # LiDAR vertical FOV range
        )
        
        # 6.2 Camera FOV mask
        camera_voxel_coord = (100, 100, 10)  # Camera position in voxel space
        occ_mask_camera = construct_surround_sensor_fov_mask(
            occ_label,
            FREE_LABEL,
            camera_voxel_coord,
            [-30, 30]  # Camera vertical FOV range
        )
        
        # 7. Process camera information
        camera_data = {}
        for i in range(4):
            camera_key = f'camera{i}'
            if camera_key in frame_data:
                camera_info = frame_data[camera_key]
                camera_data[camera_key] = {
                    'extrinsic': np.array(camera_info['extrinsic'], dtype=np.float32),
                    'intrinsic': np.array(camera_info['intrinsic'], dtype=np.float32),
                    'cords': np.array(camera_info['cords'], dtype=np.float32)
                }
            else:
                camera_data[camera_key] = {
                    'extrinsic': np.eye(4, dtype=np.float32),
                    'intrinsic': np.eye(3, dtype=np.float32),
                    'cords': np.zeros(6, dtype=np.float32)
                }
        
        if verbose:
            logging.info(f"Successfully processed frame {os.path.basename(frame_yaml_path)}")
        return (occ_label, occ_flow, occ_mask_lidar, occ_mask_camera, 
                ego_to_world_transformation_matrix, annotations, camera_data)
    
    except Exception as e:
        logging.error(f"Error processing frame {frame_yaml_path}: {str(e)}")
        return None

def save_npz(npz_path, data_tuple, verbose=False):
    """Save NPZ file"""
    try:
        (occ_label, occ_flow, occ_mask_lidar, occ_mask_camera, 
         ego_to_world_transformation_matrix, annotations, camera_data) = data_tuple
        
        save_dict = {
            'occ_label': occ_label,
            'occ_flow': occ_flow,
            'occ_mask_lidar': occ_mask_lidar,
            'occ_mask_camera': occ_mask_camera,
            'ego_to_world_transformation_matrix': ego_to_world_transformation_matrix,
            'annotations': annotations
        }
        
        for camera_key, camera_info in camera_data.items():
            save_dict[f'{camera_key}_extrinsic'] = camera_info['extrinsic']
            save_dict[f'{camera_key}_intrinsic'] = camera_info['intrinsic']
            save_dict[f'{camera_key}_cords'] = camera_info['cords']
        
        np.savez_compressed(npz_path, **save_dict)
        if verbose:
            logging.info(f"Successfully saved NPZ file: {npz_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save NPZ file {npz_path}: {str(e)}")
        return False

def process_scene(input_root, output_root, scene_name, vehicle_id, verbose=False):
    """Process data for a single scene"""
    scene_folder = os.path.join(output_root, scene_name)
    os.makedirs(scene_folder, exist_ok=True)
    
    ego_name = f"Ego{vehicle_id}"
    ego_dir = os.path.join(scene_folder, ego_name)
    os.makedirs(ego_dir, exist_ok=True)
    
    if verbose:
        logging.info(f"\nProcessing vehicle {ego_name}")
        logging.info(f"Input directory: {input_root}")
        logging.info(f"Output directory: {ego_dir}")
        
        # Check image files in input directory
        image_files = [f for f in os.listdir(input_root) if f.endswith('.png')]
        logging.info(f"Found {len(image_files)} image files in input directory")
        if image_files:
            logging.info(f"Sample image files: {image_files[:5]}")
            logging.info(f"Image naming format: {image_files[0]}")
    
    ego_info = {
        "ego_name": ego_name,
        "occ_in_scene_paths": [],
        "camera_paths": []
    }
    
    frame_counter = 1
    frames_processed = 0
    frames_skipped = 0
    images_copied = 0
    images_failed = 0
    images_missing = 0
    
    # Get all PCD files and sort them
    pcd_files = []
    for f in os.listdir(input_root):
        if f.endswith("_semantic_occluded.pcd"):
            # Verify corresponding YAML file exists
            timestamp = f.split('_')[0]
            yaml_file = f"{timestamp}.yaml"
            yaml_path = os.path.join(input_root, yaml_file)
            if os.path.exists(yaml_path):
                pcd_files.append(f)
            else:
                if verbose:
                    logging.warning(f"Skipping {f}: No corresponding YAML file {yaml_file}")
    
    pcd_files = sorted(pcd_files)
    if not pcd_files:
        logging.error(f"No valid PCD files found in {input_root}")
        return ego_info, 0, 0
    
    if verbose:
        logging.info(f"Found {len(pcd_files)} valid PCD files")
    
    # Create progress bar
    pbar = tqdm(pcd_files, desc=f"Processing {ego_name}", disable=verbose)
    
    for i, current_file in enumerate(pbar):
        timestamp = current_file.split('_')[0]
        yaml_file = f"{timestamp}.yaml"
        
        yaml_path = os.path.join(input_root, yaml_file)
        pcd_path = os.path.join(input_root, current_file)
        
        # Get next frame file paths (if they exist)
        next_yaml_path = None
        next_pcd_path = None
        if i < len(pcd_files) - 1:
            next_timestamp = pcd_files[i + 1].split('_')[0]
            next_yaml_path = os.path.join(input_root, f"{next_timestamp}.yaml")
            next_pcd_path = os.path.join(input_root, pcd_files[i + 1])
            
            # Verify next frame files exist
            if not (os.path.exists(next_yaml_path) and os.path.exists(next_pcd_path)):
                if verbose:
                    logging.warning(f"Skipping next frame {next_timestamp}: Incomplete files")
                next_yaml_path = None
                next_pcd_path = None
        
        # Process single frame
        frame_data = process_frame(yaml_path, pcd_path, next_yaml_path, next_pcd_path, verbose)
        if frame_data is None:
            if verbose:
                logging.warning(f"Skipping frame {timestamp} (incomplete files or processing failed)")
            frames_skipped += 1
            continue
        
        # Save NPZ file
        npz_filename = f"{timestamp}.npz"
        npz_path = os.path.join(ego_dir, npz_filename)
        if save_npz(npz_path, frame_data, verbose):
            relative_path = os.path.join(scene_name, ego_name, npz_filename)
            ego_info["occ_in_scene_paths"].append(relative_path)
            
            # Copy camera images and record paths
            camera_paths = []
            for j in range(4):
                camera_filename = f"{timestamp}_camera{j}.png"  # Changed from .jpg to .png
                src_camera_path = os.path.join(input_root, camera_filename)
                dst_camera_path = os.path.join(ego_dir, camera_filename)
                
                # Copy camera image
                if os.path.exists(src_camera_path):
                    try:
                        shutil.copy2(src_camera_path, dst_camera_path)
                        camera_paths.append(os.path.join(scene_name, ego_name, camera_filename))
                        images_copied += 1
                        if verbose:
                            logging.info(f"Successfully copied camera image: {camera_filename}")
                    except Exception as e:
                        logging.error(f"Failed to copy camera image {camera_filename}: {str(e)}")
                        camera_paths.append("")
                        images_failed += 1
                else:
                    if verbose:
                        logging.warning(f"Camera image does not exist: {src_camera_path}")
                    camera_paths.append("")
                    images_missing += 1
            
            ego_info["camera_paths"].append(camera_paths)
            frames_processed += 1
            if verbose:
                logging.info(f"Successfully processed frame {timestamp}")
        else:
            frames_skipped += 1
        
        frame_counter += 1
    
    if verbose:
        logging.info(f"Vehicle {ego_name} processing complete:")
        logging.info(f"- Frames processed: {frames_processed}")
        logging.info(f"- Frames skipped: {frames_skipped}")
        logging.info(f"- Images copied: {images_copied}")
        logging.info(f"- Images failed: {images_failed}")
        logging.info(f"- Images missing: {images_missing}")
    return ego_info, frames_processed, frames_skipped

def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Process OPV2V dataset and generate voxelized data')
    parser.add_argument('--input_root', type=str, required=True,
                      help='Input data root directory containing multiple scene folders')
    parser.add_argument('--output_root', type=str, required=True,
                      help='Output data root directory where CoHFF format data will be generated')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    args = parser.parse_args()
    
    # Configure paths
    input_root = args.input_root
    output_root = args.output_root
    
    # Setup logging
    log_file = setup_logging(output_root, args.verbose)
    logging.info(f"Starting data processing, log file: {log_file}")
    logging.info(f"Input directory: {input_root}")
    logging.info(f"Output directory: {output_root}")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    logging.info(f"Created output directory: {output_root}")
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)) and d.startswith("2021")]
    
    if not scene_dirs:
        logging.error(f"No scene folders found in input directory: {input_root}")
        return
    
    all_scene_infos = []
    total_scenes_processed = 0
    
    # Create progress bar for scenes
    scene_pbar = tqdm(scene_dirs, desc="Processing scenes", disable=args.verbose)
    
    for scene_dir in scene_pbar:
        scene_path = os.path.join(input_root, scene_dir)
        if args.verbose:
            logging.info(f"\nProcessing scene: {scene_dir}")
        
        # Get scene name (add "scene_" prefix)
        scene_name = f"scene_{scene_dir}"
        
        # Process all vehicles in this scene
        all_ego_info = []
        total_frames_processed = 0
        total_frames_skipped = 0
        
        # Get all vehicle directories
        vehicle_dirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
        
        for vehicle_id in vehicle_dirs:
            vehicle_dir = os.path.join(scene_path, vehicle_id)
            ego_info, frames_processed, frames_skipped = process_scene(vehicle_dir, output_root, scene_name, vehicle_id, args.verbose)
            all_ego_info.append(ego_info)
            total_frames_processed += frames_processed
            total_frames_skipped += frames_skipped
        
        # Add scene information
        scene_info = {
            "scene_name": scene_name,
            "egos": all_ego_info
        }
        all_scene_infos.append(scene_info)
        total_scenes_processed += 1
        
        if args.verbose:
            logging.info(f"\nScene {scene_name} processing complete:")
            logging.info(f"- Number of vehicles: {len(vehicle_dirs)}")
            logging.info(f"- Total frames processed: {total_frames_processed}")
            logging.info(f"- Total frames skipped: {total_frames_skipped}")
    
    # Save scene_infos.pkl
    scene_infos_path = os.path.join(output_root, "scene_infos.pkl")
    with open(scene_infos_path, "wb") as f:
        pickle.dump(all_scene_infos, f)
    
    logging.info(f"\nAll data processing complete:")
    logging.info(f"- Total scenes processed: {total_scenes_processed}")
    logging.info(f"- scene_infos.pkl saved to: {scene_infos_path}")
    logging.info(f"- Log file location: {log_file}")

if __name__ == "__main__":
    main()
