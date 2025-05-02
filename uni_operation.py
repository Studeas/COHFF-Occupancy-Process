import os
import yaml
import numpy as np
import pickle
import logging
from datetime import datetime
from pathlib import Path
import shutil
import math
from scripts.operation.crop_transform import crop_pcd, read_pcd_file, get_zdiff_egolidar, write_pcd_file
from scripts.operation.compute_occ_flow import compute_occ_flow
from scripts.operation.voxelization import voxelize_point_cloud as voxelization
import argparse
from pyquaternion import Quaternion
from tqdm import tqdm

# Constants
FREE_LABEL = 10  # Free space label

# Label mapping dictionary
OLD_OPV2V_TO_OURS_LABEL_MAPPING = {
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

OPV2V_TO_OURS_LABEL_MAPPING = {
    0: 10,  # unlabeled -> FREE
    1: 9,   # Building -> BUILDING
    2: 0,   # Fence -> GENERAL_OBJECT
    3: 0,  # unlabeled -> GENERAL_OBJECT
    4: 0,  # unlabeled -> GENERAL_OBJECT
    5: 0,   # Pole -> GENERAL_OBJECT
    6: 7,   # Road -> ROAD
    7: 7,   # Road -> ROAD
    8: 8,   # SideWalk -> WALKABLE/TERRAIN
    9: 6,   # Vegetation -> VEGETATION
    10: 1,  # Vehicles -> VEHICLE
    11: 9,  # Wall -> BUILDING
    12: 0,  # TrafficSign -> GENERAL_OBJECT
    13: 0, # unlabeled -> GENERAL_OBJECT
    14: 8,  # Ground -> WALKABLE/TERRAIN
    15: 0,  # Bridge -> GENERAL_OBJECT
    16: 0, # unlabeled -> GENERAL_OBJECT
    17: 0,  # GuardRail -> GENERAL_OBJECT
    18: 0,  # Pole -> GENERAL_OBJECT
    19: 0, # unlabeled -> GENERAL_OBJECT
    20: 1,  # Vehicles -> VEHICLE
    21: 0, # unlabeled -> GENERAL_OBJECT
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

def process_frame(frame_yaml_path, frame_pcd_path, next_yaml_path=None, next_pcd_path=None, prev_yaml_path=None, prev_pcd_path=None, verbose=False, scene_name=None, ego_name=None):
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
        z_diff = get_zdiff_egolidar(frame_yaml_path)
        if z_diff is None:
            logging.error(f"Failed to get z-value difference: {frame_yaml_path}")
            return None
            
        # Read PCD file
        header, points, labels = read_pcd_file(frame_pcd_path)
        if header is None or points is None or labels is None:
            logging.error(f"Failed to read PCD file: {frame_pcd_path}")
            return None
            
        # Crop point cloud and translate z-axis, from lidar_pose to true_ego_pos
        # cropped_points, cropped_labels = crop_pcd(points, labels, z_diff=z_diff)
        cropped_points, cropped_labels = crop_pcd(points, labels, z_diff=0) # keep the original point cloud (lidar_pose)

        if cropped_points is None or len(cropped_points) == 0:
            logging.error(f"Failed to crop and transform point cloud: {frame_pcd_path}")
            return None

        # write cropped point cloud to original pcd file
        write_pcd_file(frame_pcd_path, header, cropped_points, cropped_labels)

        #TODO 2: Voxelization
        occ_label = voxelization(frame_pcd_path)  # Use original PCD file path
        # occ_label shape: (200, 200, 16), numpy array
        if occ_label is None:
            logging.error(f"Failed to voxelize: {frame_pcd_path}")
            return None
            
        # 2.1 Remap labels
        occ_label = remap_labels(occ_label)


        # TODO 3: Compute occ_flow, occ_mask_lidar, occ_mask_camera, numpy array
        occ_flow_forward = np.zeros((200, 200, 16, 3), dtype=np.float32)
        occ_flow_backward = np.zeros((200, 200, 16, 3), dtype=np.float32)

        # Create temporary file to save voxel labels (avoid repeated creation)
        temp_npz_path = os.path.join(os.path.dirname(frame_pcd_path), "temp_voxel.npz")
        if not os.path.exists(temp_npz_path):
            np.savez(temp_npz_path, occ_label=occ_label)

        # Compute forward flow if next frame exists
        if next_yaml_path and next_pcd_path and os.path.exists(next_yaml_path) and os.path.exists(next_pcd_path):
            try:
                # Validate voxel label format
                if occ_label.shape != (200, 200, 16):
                    logging.error(f"Invalid voxel label shape: {occ_label.shape}, expected (200, 200, 16)")
                    return None

                # Log unique label values for debugging
                unique_labels = np.unique(occ_label)
                if verbose:
                    logging.info(f"Unique values in voxel labels: {unique_labels}")

                # Validate pose information in next frame YAML file using lidar_pose as ego pose
                next_frame_data = load_yaml(next_yaml_path, verbose)
                if next_frame_data is None:
                    logging.error(f"Failed to load next frame YAML file: {next_yaml_path}")
                    return None

                if 'lidar_pose' not in next_frame_data:
                    logging.error(f"Next frame YAML file missing lidar_pose: {next_yaml_path}")
                    return None

                # Compute forward flow
                dynamic_flow_forward, _, static_flow_forward, _ = compute_occ_flow(frame_yaml_path, next_yaml_path, temp_npz_path)
                
                # Validate flow shapes
                if (dynamic_flow_forward.shape != (200, 200, 16, 3) or 
                    static_flow_forward.shape != (200, 200, 16, 3)):
                    logging.error(f"Invalid forward flow shapes: dynamic={dynamic_flow_forward.shape}, static={static_flow_forward.shape}")
                    return None

                # Merge dynamic and static flows
                occ_flow_forward = dynamic_flow_forward + static_flow_forward

                if verbose:
                    logging.info("Forward flow value ranges:")
                    logging.info(f"Dynamic forward: [{np.min(dynamic_flow_forward)}, {np.max(dynamic_flow_forward)}]")
                    logging.info(f"Static forward: [{np.min(static_flow_forward)}, {np.max(static_flow_forward)}]")
                    logging.info(f"Merged forward: [{np.min(occ_flow_forward)}, {np.max(occ_flow_forward)}]")
            except Exception as e:
                logging.error(f"Failed to compute forward flow: {str(e)}")
                logging.error(f"Error details: {e.__class__.__name__}")
                import traceback
                logging.error(f"Stack trace:\n{traceback.format_exc()}")
                return None
        else:
            if verbose:
                logging.info("No next frame data, setting forward flow to 0")

        # Compute backward flow if previous frame exists
        if prev_yaml_path and prev_pcd_path and os.path.exists(prev_yaml_path) and os.path.exists(prev_pcd_path):
            try:
                # Validate pose information in previous frame YAML file using lidar_pose as ego pose
                prev_frame_data = load_yaml(prev_yaml_path, verbose)
                if prev_frame_data is None:
                    logging.error(f"Failed to load previous frame YAML file: {prev_yaml_path}")
                    return None

                if 'lidar_pose' not in prev_frame_data:
                    logging.error(f"Previous frame YAML file missing lidar_pose: {prev_yaml_path}")
                    return None

                # Compute backward flow
                # note that the compute_occ_flow function will compute the backward flow between the current frame and the previous frame
                _, dynamic_flow_backward, _, static_flow_backward = compute_occ_flow(frame_yaml_path, prev_yaml_path, temp_npz_path)

                # Validate flow shapes
                if (dynamic_flow_backward.shape != (200, 200, 16, 3) or 
                    static_flow_backward.shape != (200, 200, 16, 3)):
                    logging.error(f"Invalid backward flow shapes: dynamic={dynamic_flow_backward.shape}, static={static_flow_backward.shape}")
                    return None

                # Merge dynamic and static flows
                occ_flow_backward = dynamic_flow_backward + static_flow_backward

                if verbose:
                    logging.info("Backward flow value ranges:")
                    logging.info(f"Dynamic backward: [{np.min(dynamic_flow_backward)}, {np.max(dynamic_flow_backward)}]")
                    logging.info(f"Static backward: [{np.min(static_flow_backward)}, {np.max(static_flow_backward)}]")
                    logging.info(f"Merged backward: [{np.min(occ_flow_backward)}, {np.max(occ_flow_backward)}]")
            except Exception as e:
                logging.error(f"Failed to compute backward flow: {str(e)}")
                logging.error(f"Error details: {e.__class__.__name__}")
                import traceback
                logging.error(f"Stack trace:\n{traceback.format_exc()}")
                return None
        else:
            if verbose:
                logging.info("No previous frame data, setting backward flow to 0")

        # Delete temporary file if it exists
        if os.path.exists(temp_npz_path):
            os.remove(temp_npz_path)

            






        # 4. Extract ego_to_world_transformation from yaml(lidar frame)
        lidar_pose = frame_data.get('lidar_pose')
        if lidar_pose is None:
            logging.error(f"YAML file missing lidar_pose: {frame_yaml_path}")
            return None
            
        # Check lidar_pose format
        if not isinstance(lidar_pose, (list, tuple, np.ndarray)):
            logging.error(f"Invalid lidar_pose format, expected list or array: {type(lidar_pose)}")
            return None
            
        if len(lidar_pose) < 6:  # Check for complete 6 values
            logging.error(f"Invalid lidar_pose length, expected 6 values (position and orientation), got {len(lidar_pose)}")
            logging.error(f"lidar_pose content: {lidar_pose}")
            return None
            
        try:
            # Get position and orientation
            x, y, z = lidar_pose[:3]
            roll, pitch, yaw = lidar_pose[3:]
            if verbose:
                logging.info(f"Successfully got ego pose (lidar frame): position=[{x}, {y}, {z}], orientation=[{roll}, {pitch}, {yaw}]")
            
            # Build complete transformation matrix
            ego_to_world_transformation = np.eye(4, dtype=np.float32)
            # Set translation
            ego_to_world_transformation[:3, 3] = [x, y, z]
            # Set rotation (using Euler angles)
            # Convert angles to radians
            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            # Create rotation matrix
            q = Quaternion(axis=[1, 0, 0], angle=roll_rad) * \
                Quaternion(axis=[0, 1, 0], angle=pitch_rad) * \
                Quaternion(axis=[0, 0, 1], angle=yaw_rad)
            ego_to_world_transformation[:3, :3] = q.rotation_matrix
        except Exception as e:
            logging.error(f"Failed to process lidar_pose: {str(e)}")
            logging.error(f"lidar_pose content: {lidar_pose}")
            return None
            


        # 5. Process annotation information
        annotations = []
        if "vehicles" in frame_data:
            for token, veh in frame_data["vehicles"].items():
                try:
                    agent_locations = veh.get('location', [0, 0, 0])
                    agent_angles = veh.get('angle', [0, 0, 0])
                    agent_angles_rad = np.radians(np.array(agent_angles))
                    q_agent = (Quaternion(axis=[1, 0, 0], angle=agent_angles_rad[0]) *
                            Quaternion(axis=[0, 1, 0], angle=agent_angles_rad[1]) *
                            Quaternion(axis=[0, 0, 1], angle=agent_angles_rad[2]))
                    
                    # Calculate agent_to_ego transformation matrix using method 1
                    agent_to_ego = np.eye(4, dtype=np.float32)

                    agent_to_ego[:3, 3] = np.linalg.inv(ego_to_world_transformation[:3, :3]) @ (np.array(agent_locations) - np.array([x, y, z]))
                    agent_to_ego[:3, :3] = np.linalg.inv(ego_to_world_transformation[:3, :3]) @ q_agent.rotation_matrix
                    
                    # Calculate agent_to_world transformation matrix
                    agent_to_world = np.eye(4, dtype=np.float32)
                    agent_to_world[:3, 3] = agent_locations
                    agent_to_world[:3, :3] = q_agent.rotation_matrix
                    
                    # Get vehicle dimensions
                    agent_size = np.array(veh.get("extent", [0, 0, 0]), dtype=np.float32)
                    
                    # Set category ID (vehicle category is 1)
                    category_id = 1
                    
                    ann = {
                        "token": token,
                        "agent_to_ego": agent_to_ego,
                        "agent_to_world": agent_to_world,
                        "size": agent_size,
                        "category_id": category_id
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
        cameras = []
        camera_names = ['camera0', 'camera1', 'camera2', 'camera3']
        
        for cam_name in camera_names:
            # Get camera information from frame_data
            if cam_name in frame_data:
                camera_info = frame_data[cam_name]
                # Get camera intrinsics
                intrinsics = np.array(camera_info.get('intrinsic', np.eye(3)), dtype=np.float32)
                # Get camera extrinsics (transformation matrix relative to ego)
                extrinsics = np.array(camera_info.get('extrinsic', np.eye(4)), dtype=np.float32)
                
                # Generate image filename
                base_name = os.path.splitext(os.path.basename(frame_yaml_path))[0]
                filename = f"{scene_name}/{ego_name}/{base_name}_{cam_name}.jpg"  # 格式如: 2021_08_18_19_48_05/1045/000068_camera1.jpg
                
                cam_data = {
                    'cam_name': cam_name,
                    'filename': filename,
                    'intrinsics': intrinsics,
                    'extrinsics': extrinsics
                }
                cameras.append(cam_data)
            else:
                # Use default values if camera information is missing
                cam_data = {
                    'cam_name': cam_name,
                    'filename': '',
                    'intrinsics': np.eye(3, dtype=np.float32),
                    'extrinsics': np.eye(4, dtype=np.float32)
                }
                cameras.append(cam_data)
        
        if verbose:
            logging.info(f"Successfully processed frame {os.path.basename(frame_yaml_path)}")
        


        # 8. occ_label translation down 2 cells
        new_occ_label = np.ones_like(occ_label) * FREE_LABEL
        # new_occ_label[:, :, 2:14] = occ_label[:, :, 4:]

        new_occ_label[:, :, 2:] = occ_label[:, :, :14] # debugging
        # new_occ_label = occ_label # debugging
        
        # 8.1 occ_flow translation down 2 cells
        new_occ_flow_backward = np.zeros_like(occ_flow_backward)
        new_occ_flow_forward = np.zeros_like(occ_flow_forward)
        # new_occ_flow_backward[:, :, 2:14, :] = occ_flow_backward[:, :, 4:, :]
        # new_occ_flow_forward[:, :, 2:14, :] = occ_flow_forward[:, :, 4:, :]

        new_occ_flow_backward[:, :, 2:, :] = occ_flow_backward[:, :, :14, :] # debugging
        new_occ_flow_forward[:, :, 2:, :] = occ_flow_forward[:, :, :14, :] # debugging

        # new_occ_flow_backward = occ_flow_backward # debugging
        # new_occ_flow_forward = occ_flow_forward # debugging

        
        
        
        
        return (new_occ_label, new_occ_flow_forward, new_occ_flow_backward, occ_mask_lidar, occ_mask_camera, 
                ego_to_world_transformation, annotations, cameras)
    
    except Exception as e:
        logging.error(f"Error processing frame {frame_yaml_path}: {str(e)}")
        return None

def save_npz(data_tuple, npz_path, verbose=False):
    """Save NPZ file"""
    try:
        (occ_label, occ_flow_forward, occ_flow_backward, occ_mask_lidar, occ_mask_camera, ego_to_world_transformation, annotations, cameras) = data_tuple
        
        save_dict = {
            'occ_label': occ_label,
            'occ_flow_forward': occ_flow_forward,
            'occ_flow_backward': occ_flow_backward,
            'occ_mask_lidar': occ_mask_lidar,
            'occ_mask_camera': occ_mask_camera,
            'ego_to_world_transformation': ego_to_world_transformation,
            'annotations': annotations,
            "cameras": cameras
        }
        
        # for cam_idx, cam_data in enumerate(cameras):
        #     save_dict[f'camera{cam_idx}_name'] = cam_data['cam_name']
        #     save_dict[f'camera{cam_idx}_filename'] = cam_data['filename']
        #     save_dict[f'camera{cam_idx}_intrinsics'] = cam_data['intrinsics']
        #     save_dict[f'camera{cam_idx}_extrinsics'] = cam_data['extrinsics']
        
        np.savez_compressed(npz_path, **save_dict)
        if verbose:
            logging.info(f"Successfully saved NPZ file: {npz_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save NPZ file {npz_path}: {str(e)}")
        return False

def process_scene_vehicle(vehicle_dir, output_dir, verbose=False):
    """Process data for a single scene"""
    # Get scene name and ego name
    scene_name = f"{os.path.basename(os.path.dirname(vehicle_dir))}"  # e.g. 2021_08_20_21_48_35
    ego_name = f"{os.path.basename(vehicle_dir)}"  # e.g. 2149

    # Create necessary directories
    ego_dir = os.path.join(output_dir, scene_name, ego_name)
    os.makedirs(ego_dir, exist_ok=True)
    
    if verbose:
        logging.info(f"Processing scene: {vehicle_dir}")
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(vehicle_dir) if f.endswith('.png')]
    if verbose:
        logging.info(f"Found {len(image_files)} image files")
    
    yaml_files_names = [f for f in os.listdir(vehicle_dir) if f.endswith('.yaml')]
    if verbose:
        logging.info(f"Found {len(yaml_files_names)} YAML files")
    
    yaml_files = [os.path.join(vehicle_dir, f) for f in yaml_files_names]

    # Collect PCD files and their corresponding YAML files
    semantic_pcd_files = [] # only take semantic_occluded.pcd i

    for yaml_file in yaml_files_names:
        base_name = os.path.splitext(yaml_file)[0]
        semantic_pcd_file = os.path.join(vehicle_dir, f"{base_name}_semantic_occluded.pcd")
        if os.path.exists(semantic_pcd_file):
            semantic_pcd_files.append(semantic_pcd_file)

    if verbose:
        logging.info(f"Found {len(semantic_pcd_files)} valid PCD-YAML pairs")
    
    # Process each frame
    frames_processed = 0
    frames_skipped = 0

    ego_info = {
        "ego_name": ego_name, # e.g. Ego2149
        "occ_in_scene_paths": [] # every frame npz file path in this scene
    }
    
    # 检查已处理的帧
    processed_frames = set()
    for file in os.listdir(ego_dir):
        if file.endswith('.npz'):
            base_name = os.path.splitext(file)[0]
            processed_frames.add(base_name)
            # 添加已处理的帧路径
            relative_npz_path = os.path.join(scene_name, ego_name, f"{base_name}.npz")
            ego_info["occ_in_scene_paths"].append(relative_npz_path)
            frames_processed += 1
    
    if verbose:
        logging.info(f"Found {len(processed_frames)} already processed frames")
    
    for i in tqdm(range(len(semantic_pcd_files)), desc="Processing frames"):
        frame_yaml_path = yaml_files[i]
        frame_pcd_path = semantic_pcd_files[i]
        base_name = os.path.splitext(os.path.basename(frame_yaml_path))[0]
        
        # 跳过已处理的帧
        if base_name in processed_frames:
            if verbose:
                logging.info(f"Skipping already processed frame: {base_name}")
            continue
        
        # Get previous frame paths if not the first frame
        prev_yaml_path = None
        prev_pcd_path = None
        if i > 0:
            prev_yaml_path = yaml_files_names[i-1]
            prev_pcd_path = semantic_pcd_files[i-1]
        
        # Get next frame paths if not the last frame
        next_yaml_path = None
        next_pcd_path = None
        if i < len(semantic_pcd_files) - 1:
            next_yaml_path = yaml_files[i+1]
            next_pcd_path = semantic_pcd_files[i+1]
        
        # core process
        result = process_frame(
            frame_yaml_path, 
            frame_pcd_path,
            next_yaml_path=next_yaml_path,
            next_pcd_path=next_pcd_path,
            prev_yaml_path=prev_yaml_path,
            prev_pcd_path=prev_pcd_path,
            verbose=verbose,
            scene_name=scene_name,
            ego_name=ego_name
        )
        
        if result is None:
            logging.error(f"Failed to process frame {os.path.basename(frame_yaml_path)}")
            frames_skipped += 1
            continue
            
        # Save processed data
        npz_path = os.path.join(ego_dir, f"{base_name}.npz")
        relative_npz_path = os.path.join(scene_name, ego_name, f"{base_name}.npz")
        
        try:
            save_npz(result, npz_path) # take absolute path as input
            if verbose:
                logging.info(f"Successfully saved NPZ file: {npz_path}")
            
            # Add frame information
            ego_info["occ_in_scene_paths"].append(relative_npz_path)
            frames_processed += 1
        except Exception as e:
            logging.error(f"Failed to save NPZ file {npz_path}: {str(e)}")
            frames_skipped += 1
            continue
            
        # Copy camera images
        img_paths = [os.path.join(vehicle_dir, f"{base_name}_camera{i}.png") for i in range(4)]
        for img_path in img_paths:
            if os.path.exists(img_path):
                try:
                    shutil.copy2(img_path, os.path.join(ego_dir, os.path.basename(img_path)))
                    if verbose:
                        logging.info(f"Successfully copied image: {img_path} to {ego_dir}")
                except Exception as e:
                    logging.error(f"Failed to copy image {img_path}: {str(e)}")
                    continue
            else:
                logging.warning(f"Image file not found: {img_path}")
    
    if verbose:
        logging.info(f"Finished processing scene: {vehicle_dir}")
    
    # ego_info is lastly used for scene_infos.pkl
    return ego_info, frames_processed, frames_skipped


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Process OPV2V dataset and generate voxelized data')
    parser.add_argument('--input_root', type=str, required=True, help='Input data root directory containing multiple scene folders')
    parser.add_argument('--output_root', type=str, required=True, help='Output data root directory where CoHFF format data will be generated')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # 配置路径
    input_root = args.input_root
    output_root = args.output_root
    verbose = args.verbose
    
    # 设置日志
    log_file = setup_logging(output_root, verbose)
    logging.info(f"Starting data processing, log file: {log_file}")
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 获取所有场景目录
    vehicle_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)) and d.startswith("2021")]
    
    if not vehicle_dirs:
        logging.error(f"No scene folders found in input directory: {input_root}")
        return
    
    all_scene_infos = []
    total_scenes_processed = 0
    
    # 检查已处理的场景
    processed_scenes = set()
    if os.path.exists(os.path.join(output_root, "scene_infos.pkl")):
        try:
            with open(os.path.join(output_root, "scene_infos.pkl"), 'rb') as f:
                existing_scene_infos = pickle.load(f)
                for scene_info in existing_scene_infos:
                    processed_scenes.add(scene_info["scene_name"])
                    all_scene_infos.append(scene_info)
                    total_scenes_processed += 1
        except Exception as e:
            logging.error(f"Error loading existing scene_infos.pkl: {str(e)}")
    
    # 计算剩余需要处理的场景
    remaining_scenes = [d for d in vehicle_dirs if d not in processed_scenes]
    total_remaining_scenes = len(remaining_scenes)
    
    if total_remaining_scenes == 0:
        logging.info("All scenes have been processed. No new scenes to process.")
        return
    
    # 创建场景处理进度条
    scene_pbar = tqdm(vehicle_dirs, desc="Processing scenes")
    
    # 添加时间统计
    import time
    start_time = time.time()
    last_save_time = start_time
    save_interval = 300  # 每5分钟保存一次进度
    
    for current_scene_dir in scene_pbar:
        if current_scene_dir in processed_scenes:
            continue
            
        scene_path = os.path.join(input_root, current_scene_dir)
        scene_name = current_scene_dir
        
        # 处理该场景中的所有车辆
        all_ego_info = []
        total_frames_processed = 0
        total_frames_skipped = 0
        
        # 获取所有车辆目录
        vehicle_dirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
        
        for vehicle_id in vehicle_dirs:
            current_scene_dir = os.path.join(scene_path, vehicle_id)
            ego_info, frames_processed, frames_skipped = process_scene_vehicle(current_scene_dir, output_root, verbose)
            all_ego_info.append(ego_info)
            total_frames_processed += frames_processed
            total_frames_skipped += frames_skipped
        
        # 添加场景信息
        scene_info = {
            "scene_name": scene_name,
            "egos": all_ego_info
        }
        all_scene_infos.append(scene_info)
        total_scenes_processed += 1
        
        # 定期保存进度
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            scene_infos_path = os.path.join(output_root, "scene_infos.pkl")
            with open(scene_infos_path, "wb") as f:
                pickle.dump(all_scene_infos, f)
            last_save_time = current_time
    
    # 最终保存
    scene_infos_path = os.path.join(output_root, "scene_infos.pkl")
    with open(scene_infos_path, "wb") as f:
        pickle.dump(all_scene_infos, f)
    
    total_time = time.time() - start_time
    logging.info(f"\nProcessing complete:")
    logging.info(f"- Total scenes processed: {total_scenes_processed}")
    logging.info(f"- Total time: {total_time/3600:.2f} hours")
    logging.info(f"- Log file: {log_file}")


def debugger_main():
    # 硬编码的路径参数
    # input_root = "D:/semantic(testvalidate)/test"
    # output_root = "D:/COHFF-test"
    input_root = "C:/semanticlidar_18_unzip/train" #"D:/semantic(testvalidate)/validate"
    output_root = "D:/COHFF-train" #"D:/COHFF-validate"
    verbose = True  # 设置为True以启用详细日志
    
    # 设置日志
    log_file = setup_logging(output_root, verbose)
    logging.info(f"Starting data processing, log file: {log_file}")
    logging.info(f"Input directory: {input_root}")
    logging.info(f"Output directory: {output_root}")
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    logging.info(f"Created output directory: {output_root}")
    
    # 获取所有场景目录
    vehicle_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)) and d.startswith("2021")]
    
    if not vehicle_dirs:
        logging.error(f"No scene folders found in input directory: {input_root}")
        return
    
    all_scene_infos = []
    total_scenes_processed = 0
    
    # 检查已处理的场景
    processed_scenes = set()
    if os.path.exists(os.path.join(output_root, "scene_infos.pkl")):
        try:
            with open(os.path.join(output_root, "scene_infos.pkl"), 'rb') as f:
                existing_scene_infos = pickle.load(f)
                for scene_info in existing_scene_infos:
                    processed_scenes.add(scene_info["scene_name"])
                    all_scene_infos.append(scene_info)
                    total_scenes_processed += 1
            logging.info(f"Found {len(processed_scenes)} already processed scenes")
        except Exception as e:
            logging.error(f"Error loading existing scene_infos.pkl: {str(e)}")
    
    # 计算剩余需要处理的场景数量
    remaining_scenes = [d for d in vehicle_dirs if d not in processed_scenes]
    total_remaining_scenes = len(remaining_scenes)
    
    if total_remaining_scenes == 0:
        logging.info("All scenes have been processed. No new scenes to process.")
        return
    
    logging.info(f"Total scenes to process: {total_remaining_scenes}")
    
    # 创建场景处理进度条
    scene_pbar = tqdm(vehicle_dirs, desc="Processing scenes")
    
    # 添加时间统计
    import time
    start_time = time.time()
    last_save_time = start_time
    save_interval = 300  # 每5分钟保存一次进度
    last_log_time = start_time
    log_interval = 60  # 每分钟更新一次进度
    
    for current_scene_dir in scene_pbar:
        # 跳过已处理的场景
        if current_scene_dir in processed_scenes:
            if verbose:
                logging.info(f"Skipping already processed scene: {current_scene_dir}")
            continue
            
        scene_path = os.path.join(input_root, current_scene_dir)
        scene_name = current_scene_dir
        if verbose:
            logging.info(f"\nProcessing scene: {current_scene_dir}")
        
        # 处理该场景中的所有车辆
        all_ego_info = []
        total_frames_processed = 0
        total_frames_skipped = 0
        
        # 获取所有车辆目录
        vehicle_dirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
        
        for vehicle_id in vehicle_dirs:
            current_scene_dir = os.path.join(scene_path, vehicle_id)

            # 核心处理
            ego_info, frames_processed, frames_skipped = process_scene_vehicle(current_scene_dir, output_root, verbose)
            all_ego_info.append(ego_info)
            total_frames_processed += frames_processed
            total_frames_skipped += frames_skipped
        
        # 添加场景信息（用于pkl文件）
        scene_info = {
            "scene_name": scene_name,
            "egos": all_ego_info
        }
        all_scene_infos.append(scene_info)
        total_scenes_processed += 1
        
        # 计算处理时间和剩余时间
        current_time = time.time()
        elapsed_time = current_time - start_time
        scenes_per_second = total_scenes_processed / elapsed_time if elapsed_time > 0 else 0
        remaining_time = (total_remaining_scenes - total_scenes_processed) / scenes_per_second if scenes_per_second > 0 else 0
        
        # 定期更新进度信息
        if current_time - last_log_time >= log_interval:
            logging.info(f"\nProgress Update:")
            logging.info(f"- Scenes processed: {total_scenes_processed}/{total_remaining_scenes}")
            logging.info(f"- Processing speed: {scenes_per_second*3600:.2f} scenes/hour")
            logging.info(f"- Elapsed time: {elapsed_time/3600:.2f} hours")
            logging.info(f"- Estimated remaining time: {remaining_time/3600:.2f} hours")
            last_log_time = current_time
        
        # 定期保存进度
        if current_time - last_save_time >= save_interval:
            scene_infos_path = os.path.join(output_root, "scene_infos.pkl")
            with open(scene_infos_path, "wb") as f:
                pickle.dump(all_scene_infos, f)
            last_save_time = current_time
            logging.info(f"Progress saved at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if verbose:
            logging.info(f"\nScene {scene_name} processing complete:")
            logging.info(f"- Number of vehicles: {len(vehicle_dirs)}")
            logging.info(f"- Total frames processed: {total_frames_processed}")
            logging.info(f"- Total frames skipped: {total_frames_skipped}")
            logging.info(f"- Processing speed: {scenes_per_second*3600:.2f} scenes/hour")
            logging.info(f"- Estimated remaining time: {remaining_time/3600:.2f} hours")
    
    # 最终保存
    scene_infos_path = os.path.join(output_root, "scene_infos.pkl")
    with open(scene_infos_path, "wb") as f:
        pickle.dump(all_scene_infos, f)
    
    total_time = time.time() - start_time
    logging.info(f"\nAll data processing complete:")
    logging.info(f"- Total scenes processed: {total_scenes_processed}")
    logging.info(f"- Total processing time: {total_time/3600:.2f} hours")
    logging.info(f"- Average processing speed: {total_scenes_processed/total_time*3600:.2f} scenes/hour")
    logging.info(f"- scene_infos.pkl saved to: {scene_infos_path}")
    logging.info(f"- Log file location: {log_file}")


if __name__ == "__main__":
    # debugger_main()
    main()
