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


def compute_flow(current_yaml_path, target_yaml_path, voxel_npz_path, verbose=False):
    """
    Compute flow (flow field) between adjacent frames using vectorized operations.

    Args:
        current_yaml_path: Path to current frame YAML file.
        next_yaml_path: Path to next frame YAML file.
        voxel_npz_path: Path to voxel NPZ file.
        verbose: Whether to output detailed logs.

    Returns:
        Tuple: (dynamic_flow_forward, dynamic_flow_backward, static_flow_forward, static_flow_backward)
        Each array has shape (200, 200, 16, 3).
    """
    try:
        # Load YAML data for current and next frames
        current_data = read_yaml_file(current_yaml_path, verbose)
        next_data = read_yaml_file(target_yaml_path, verbose)
        if current_data is None or next_data is None:
            return None, None, None, None

        # Check and extract lidar_pose information (using lidar_pose as ego pose)
        if 'lidar_pose' not in current_data or len(current_data['lidar_pose']) < 6:
            if verbose:
                logging.error(f"Error: Current YAML file missing or invalid lidar_pose. File: {current_yaml_path}")
            return None, None, None, None
        if 'lidar_pose' not in next_data or len(next_data['lidar_pose']) < 6:
            if verbose:
                logging.error(f"Error: Next YAML file missing or invalid lidar_pose. File: {target_yaml_path}")
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
        occ_label = voxel_data['occ_label']  # 虽然不作为条件过滤，每个体素都要计算流向

        # 生成整个体素网格物理坐标 (200,200,16,3)
        xs = (np.arange(200) - 100) * 0.4
        ys = (np.arange(200) - 100) * 0.4
        zs = (np.arange(16) - 8) * 0.4
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
        coords = np.stack([grid_x, grid_y, grid_z], axis=-1)  # (200,200,16,3)

        # 扩展为齐次坐标，形状 (200,200,16,4)
        ones = np.ones_like(grid_x)
        coords_hom = np.concatenate([coords, ones[..., np.newaxis]], axis=-1)
        N = 200 * 200 * 16
        coords_hom_flat = coords_hom.reshape(N, 4)  # (N,4)
        coords_xyz_flat = coords_hom_flat[:, :3]  # (N,3)

        # 准备静态流结果数组，先全部初始化为0
        static_flow_forward_flat = np.zeros((N, 3), dtype=np.float32)
        static_flow_backward_flat = np.zeros((N, 3), dtype=np.float32)

        # 创建非 FREE 区域的掩码
        FREE_LABEL = 10
        occ_label_flat = occ_label.reshape(N)
        non_free_mask = occ_label_flat != FREE_LABEL


        # # 向量化计算静态流
        # static_forward_coords = (static_forward_transform @ coords_hom_flat.T).T  # (N,4)
        # static_backward_coords = (static_backward_transform @ coords_hom_flat.T).T  # (N,4)
        # static_flow_forward_flat = static_forward_coords[:, :3] - coords_xyz_flat  # (N,3)
        # static_flow_backward_flat = static_backward_coords[:, :3] - coords_xyz_flat  # (N,3)

        # 仅对非 FREE 的体素计算 static flow
        if np.any(non_free_mask):
            # 计算正向 static flow
            static_forward_coords = (static_forward_transform @ coords_hom_flat[non_free_mask].T).T  # (n,4)
            static_flow_forward_flat[non_free_mask] = static_forward_coords[:, :3] - coords_xyz_flat[non_free_mask]

            # 计算反向 static flow
            static_backward_coords = (static_backward_transform @ coords_hom_flat[non_free_mask].T).T  # (n,4)
            static_flow_backward_flat[non_free_mask] = static_backward_coords[:, :3] - coords_xyz_flat[non_free_mask]

        # 将静态流重新 reshape 成 (200,200,16,3)
        static_flow_forward = static_flow_forward_flat.reshape(200, 200, 16, 3)
        static_flow_backward = static_flow_backward_flat.reshape(200, 200, 16, 3)

        # 初始化动态流（全部为0，后续对车辆区域进行更新），形状 (N,3)
        dynamic_flow_forward_flat = np.zeros((N, 3), dtype=np.float32)
        dynamic_flow_backward_flat = np.zeros((N, 3), dtype=np.float32)
        dynamic_assigned = np.zeros(N, dtype=bool)

        # 提取车辆注释：仅处理同时在当前帧与下一帧中出现的车辆
        current_vehicles = current_data.get('vehicles', {})
        next_vehicles = next_data.get('vehicles', {})
        common_vehicle_tokens = set(current_vehicles.keys()).intersection(set(next_vehicles.keys()))

        # 预先计算每个车辆的动态变换矩阵（正向和反向）及车辆当前位置信息
        vehicle_dyn_forward = {}
        vehicle_dyn_backward = {}
        veh_locations = {}
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

            # 预先计算动态转换矩阵
            dyn_forward = np.linalg.inv(T_e_next) @ T_v_next @ np.linalg.inv(T_v_current) @ T_e_current
            dyn_backward = np.linalg.inv(T_e_current) @ T_v_current @ np.linalg.inv(T_v_next) @ T_e_next
            vehicle_dyn_forward[token] = dyn_forward
            vehicle_dyn_backward[token] = dyn_backward

        # 对每个车辆，根据其位置生成布尔 mask，然后批量计算动态流
        for token in common_vehicle_tokens:
            veh_loc = veh_locations[token]  # 当前帧车辆位置，形状 (3,)
            # 计算所有体素到该车辆位置的距离，形状 (N,)
            dists = np.linalg.norm(coords_xyz_flat - veh_loc, axis=1)
            mask = dists < 5.0  # 阈值5米
            # 仅更新那些尚未分配过动态流的体素
            mask = mask & (~dynamic_assigned)
            if np.any(mask):
                # 正向动态流
                dyn_forward = vehicle_dyn_forward[token]
                new_coords_forward = (dyn_forward @ coords_hom_flat[mask].T).T  # (n,4)
                dyn_flow_forward = new_coords_forward[:, :3] - coords_xyz_flat[mask]
                dynamic_flow_forward_flat[mask] = dyn_flow_forward
                # 反向动态流
                dyn_backward = vehicle_dyn_backward[token]
                new_coords_backward = (dyn_backward @ coords_hom_flat[mask].T).T
                dyn_flow_backward = new_coords_backward[:, :3] - coords_xyz_flat[mask]
                dynamic_flow_backward_flat[mask] = dyn_flow_backward
                dynamic_assigned[mask] = True

        # 将动态流重新 reshape 成 (200,200,16,3)
        dynamic_flow_forward = dynamic_flow_forward_flat.reshape(200, 200, 16, 3)
        dynamic_flow_backward = dynamic_flow_backward_flat.reshape(200, 200, 16, 3)

        if verbose:
            logging.info(f"Successfully computed flows with shapes: dynamic_forward={dynamic_flow_forward.shape}, "
                         f"dynamic_backward={dynamic_flow_backward.shape}, static_forward={static_flow_forward.shape}, "
                         f"static_backward={static_flow_backward.shape}")
        return dynamic_flow_forward, dynamic_flow_backward, static_flow_forward, static_flow_backward

    except Exception as e:
        logging.error(f"Error computing flow: {str(e)}")
        return None, None, None, None

def compute_occ_flow(current_yaml_path, target_yaml_path, voxel_npz_path, verbose=False):
    """
    Compute occupancy flow between adjacent frames.
    """
    result = compute_flow(current_yaml_path, target_yaml_path, voxel_npz_path, verbose)
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
