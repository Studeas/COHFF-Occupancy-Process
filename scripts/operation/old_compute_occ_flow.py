import numpy as np
import yaml
import os
import logging
from pyquaternion import Quaternion
from tqdm import tqdm  # 用于显示进度条

def read_yaml_file(yaml_path, verbose=False):
    """Read YAML file and return parsed data"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        if verbose:
            logging.error(f"Error processing YAML file {yaml_path}: {str(e)}")
        return None

def compute_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose=False):
    """
    Compute flow (flow field) between adjacent frames

    Args:
        current_yaml_path: Path to current frame YAML file
        next_yaml_path: Path to next frame YAML file
        voxel_npz_path: Path to voxel NPZ file
        verbose: Whether to output detailed logs

    Returns:
        Tuple: (dynamic_flow_forward, dynamic_flow_backward, static_flow_forward, static_flow_backward)
        Each array has shape (200, 200, 16, 3)
    """
    try:
        # Load YAML data for current and next frames
        current_data = read_yaml_file(current_yaml_path, verbose)
        next_data = read_yaml_file(next_yaml_path, verbose)
        if current_data is None or next_data is None:
            return None, None, None, None

        # Check and extract lidar_pose information
        if 'lidar_pose' not in current_data or len(current_data['lidar_pose']) < 6:
            if verbose:
                logging.error(f"Error: Current YAML file missing or invalid lidar_pose. File: {current_yaml_path}")
            return None, None, None, None
        if 'lidar_pose' not in next_data or len(next_data['lidar_pose']) < 6:
            if verbose:
                logging.error(f"Error: Next YAML file missing or invalid lidar_pose. File: {next_yaml_path}")
            return None, None, None, None

        current_ego_pose = current_data['lidar_pose']
        next_ego_pose = next_data['lidar_pose']

        # Extract position and angles, convert to radians
        current_pos = np.array(current_ego_pose[:3])
        current_rot = np.radians(np.array(current_ego_pose[3:]))
        next_pos = np.array(next_ego_pose[:3])
        next_rot = np.radians(np.array(next_ego_pose[3:]))

        # Build ego transformation matrices
        current_q = (Quaternion(axis=[1, 0, 0], angle=current_rot[0]) *
                     Quaternion(axis=[0, 1, 0], angle=current_rot[1]) *
                     Quaternion(axis=[0, 0, 1], angle=current_rot[2]))
        next_q = (Quaternion(axis=[1, 0, 0], angle=next_rot[0]) *
                  Quaternion(axis=[0, 1, 0], angle=next_rot[1]) *
                  Quaternion(axis=[0, 0, 1], angle=next_rot[2]))

        T_e_current = np.eye(4)
        T_e_current[:3, :3] = current_q.rotation_matrix
        T_e_current[:3, 3] = current_pos

        T_e_next = np.eye(4)
        T_e_next[:3, :3] = next_q.rotation_matrix
        T_e_next[:3, 3] = next_pos

        # Precompute static (ego-only) transforms
        static_forward_transform = np.linalg.inv(T_e_next) @ T_e_current
        static_backward_transform = np.linalg.inv(T_e_current) @ T_e_next

        # Load voxel data
        voxel_data = np.load(voxel_npz_path)
        occ_label = voxel_data['occ_label']

        # Initialize flow field arrays
        dynamic_flow_forward = np.zeros((200, 200, 16, 3), dtype=np.float32)
        dynamic_flow_backward = np.zeros((200, 200, 16, 3), dtype=np.float32)
        static_flow_forward = np.zeros((200, 200, 16, 3), dtype=np.float32)
        static_flow_backward = np.zeros((200, 200, 16, 3), dtype=np.float32)

        # Extract vehicle annotation information from YAML data (current and next frames)
        current_vehicles = current_data.get('vehicles', {})
        next_vehicles = next_data.get('vehicles', {})

        # 预先计算同时出现在两个帧中的车辆的变换矩阵及其位置，避免重复计算
        common_vehicle_tokens = set(current_vehicles.keys()).intersection(set(next_vehicles.keys()))
        veh_locations = {}  # 当前帧车辆位置，用于距离判断
        vehicle_dyn_forward = {}  # 预先计算的动态正向变换
        vehicle_dyn_backward = {} # 预先计算的动态反向变换
        for token in common_vehicle_tokens:
            veh_curr = current_vehicles[token]
            veh_next = next_vehicles[token]

            # 当前帧车辆变换矩阵
            curr_angle = np.radians(np.array(veh_curr.get('angle', [0, 0, 0])))
            q_v_current = (Quaternion(axis=[1, 0, 0], angle=curr_angle[0]) *
                           Quaternion(axis=[0, 1, 0], angle=curr_angle[1]) *
                           Quaternion(axis=[0, 0, 1], angle=curr_angle[2]))
            T_v_current = np.eye(4)
            T_v_current[:3, :3] = q_v_current.rotation_matrix
            T_v_current[:3, 3] = np.array(veh_curr.get('location', [0, 0, 0]))

            # 下一帧车辆变换矩阵
            next_angle = np.radians(np.array(veh_next.get('angle', [0, 0, 0])))
            q_v_next = (Quaternion(axis=[1, 0, 0], angle=next_angle[0]) *
                        Quaternion(axis=[0, 1, 0], angle=next_angle[1]) *
                        Quaternion(axis=[0, 0, 1], angle=next_angle[2]))
            T_v_next = np.eye(4)
            T_v_next[:3, :3] = q_v_next.rotation_matrix
            T_v_next[:3, 3] = np.array(veh_next.get('location', [0, 0, 0]))

            veh_locations[token] = np.array(veh_curr.get('location', [0, 0, 0]))

            # 预先计算动态变换矩阵（正向和反向）
            dyn_forward = np.linalg.inv(T_e_next) @ T_v_next @ np.linalg.inv(T_v_current) @ T_e_current
            dyn_backward = np.linalg.inv(T_e_current) @ T_v_current @ np.linalg.inv(T_v_next) @ T_e_next
            vehicle_dyn_forward[token] = dyn_forward
            vehicle_dyn_backward[token] = dyn_backward

        # 遍历整个 200×200×16 的体素网格（遍历所有体素）
        total_voxels = 200 * 200 * 16
        for idx in tqdm(np.ndindex((200, 200, 16)), total=total_voxels, desc="Processing voxels"):
            x, y, z = idx
            # 将体素索引转换为以 ego 坐标系表示的物理坐标（单位：米）
            voxel_pos = np.array([
                (x - 100) * 0.4,
                (y - 100) * 0.4,
                (z - 8) * 0.4
            ])
            voxel_hom = np.append(voxel_pos, 1)  # 扩充为齐次坐标

            # 判断该体素是否在某车辆附近（阈值5米），遍历所有预先计算的车辆
            is_vehicle = False
            vehicle_token = None
            for token, veh_loc in veh_locations.items():
                if np.linalg.norm(voxel_pos - veh_loc) < 5.0:
                    is_vehicle = True
                    vehicle_token = token
                    break

            if is_vehicle:
                # 使用预先计算的车辆动态变换矩阵
                dyn_forward = vehicle_dyn_forward[vehicle_token]
                dyn_backward = vehicle_dyn_backward[vehicle_token]

                # 正向动态流
                next_voxel_hom = dyn_forward @ voxel_hom
                next_voxel = next_voxel_hom[:3]
                next_x = int(next_voxel[0] / 0.4 + 100)
                next_y = int(next_voxel[1] / 0.4 + 100)
                next_z = int(next_voxel[2] / 0.4 + 8)
                if 0 <= next_x < 200 and 0 <= next_y < 200 and 0 <= next_z < 16:
                    dynamic_flow_forward[x, y, z] = next_voxel - voxel_pos

                # 反向动态流
                prev_voxel_hom = dyn_backward @ voxel_hom
                prev_voxel = prev_voxel_hom[:3]
                prev_x = int(prev_voxel[0] / 0.4 + 100)
                prev_y = int(prev_voxel[1] / 0.4 + 100)
                prev_z = int(prev_voxel[2] / 0.4 + 8)
                if 0 <= prev_x < 200 and 0 <= prev_y < 200 and 0 <= prev_z < 16:
                    dynamic_flow_backward[x, y, z] = prev_voxel - voxel_pos
            else:
                # 对于静态物体，仅考虑ego运动
                next_voxel_hom = static_forward_transform @ voxel_hom
                next_voxel = next_voxel_hom[:3]
                next_x = int(next_voxel[0] / 0.4 + 100)
                next_y = int(next_voxel[1] / 0.4 + 100)
                next_z = int(next_voxel[2] / 0.4 + 8)
                if 0 <= next_x < 200 and 0 <= next_y < 200 and 0 <= next_z < 16:
                    static_flow_forward[x, y, z] = next_voxel - voxel_pos

                prev_voxel_hom = static_backward_transform @ voxel_hom
                prev_voxel = prev_voxel_hom[:3]
                prev_x = int(prev_voxel[0] / 0.4 + 100)
                prev_y = int(prev_voxel[1] / 0.4 + 100)
                prev_z = int(prev_voxel[2] / 0.4 + 8)
                if 0 <= prev_x < 200 and 0 <= prev_y < 200 and 0 <= prev_z < 16:
                    static_flow_backward[x, y, z] = prev_voxel - voxel_pos

        if verbose:
            logging.info(f"Successfully computed flows with shapes: dynamic_forward={dynamic_flow_forward.shape}, "
                         f"dynamic_backward={dynamic_flow_backward.shape}, static_forward={static_flow_forward.shape}, "
                         f"static_backward={static_flow_backward.shape}")
        return dynamic_flow_forward, dynamic_flow_backward, static_flow_forward, static_flow_backward

    except Exception as e:
        logging.error(f"Error computing flow: {str(e)}")
        return None, None, None, None

def compute_occ_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose=False):
    """
    Compute occupancy flow between adjacent frames
    """
    result = compute_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose)
    return result

if __name__ == "__main__":
    # Set input file paths
    current_yaml_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045\000000.yaml"
    next_yaml_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045\000001.yaml"
    voxel_npz_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045_cropped_voxel\000000_voxel.npz"

    # Compute flow
    dynamic_flow_forward, dynamic_flow_backward, static_flow_forward, static_flow_backward = \
        compute_occ_flow(current_yaml_path, next_yaml_path, voxel_npz_path, verbose=True)

    if dynamic_flow_forward is not None and dynamic_flow_backward is not None and \
       static_flow_forward is not None and static_flow_backward is not None:
        logging.info(f"Successfully computed flows with shapes: dynamic_forward={dynamic_flow_forward.shape}, "
                     f"dynamic_backward={dynamic_flow_backward.shape}, static_forward={static_flow_forward.shape}, "
                     f"static_backward={static_flow_backward.shape}")
    else:
        logging.error("Failed to compute flow")
