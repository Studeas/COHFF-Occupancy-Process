import open3d as o3d # type: ignore
import numpy as np
import math
from typing import Tuple, List, Dict, Iterable
import argparse

# Constants
NOT_OBSERVED = -1
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0
OCCUPIED = 1
FREE_LABEL = 10
MAX_POINT_NUM = 10
LABEL_ROAD = 7
VOXEL_SIZE = [0.4, 0.4, 0.4]
SPTIAL_SHAPE = [1600, 1600, 64]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
VIS = False
FILL_ROAD = False


# Weak
# COLOR_MAP = np.array([
#     [255, 255, 255, 255],  # 0 undefined (almost white)
#     [200, 220, 255, 255],  # 1 car             very light blue
#     [255, 240, 245, 255],  # 2 bicycle         very light pink
#     [245, 230, 170, 255],  # 3 motorcycle      very light orange
#     [255, 180, 180, 255],  # 4 pedestrian      very light red
#     [255, 255, 230, 255],  # 5 traffic_cone    very light yellow
#     [220, 240, 220, 255],  # 6 vegetation      very light green
#     [255, 200, 255, 255],  # 7 road            very light pink
#     [220, 240, 220, 255],  # 8 terrain         very light green
#     [250, 250, 255, 255],  # 9 building        even lighter white
#     [0, 0, 0, 0],          # 10 Unobserved (unchanged)
#     [230, 230, 230, 255],  # 11 Visualized voxel (very light gray)
#     [0, 150, 245, 255],         # 1 car             blue
#     [255, 192, 203, 255],       # 2 bicycle         pink
#     [200, 180, 0, 255],         # 3 motorcycle      dark orange
#     [255, 0, 0, 255],           # 4 pedestrian      red
#     [255, 240, 150, 255],       # 5 traffic_cone    light yellow
#     [0, 175, 0, 255],           # 6 vegetation      green
#     [255, 0, 255, 255],         # 7 road            dark pink
#     [0, 175, 0, 255],           # 8 terrain         green    [0, 150, 245, 255],         # 1 car             blue
#     [255, 192, 203, 255],       # 2 bicycle         pink
#     [200, 180, 0, 255],         # 3 motorcycle      dark orange
#     [255, 0, 0, 255],           # 4 pedestrian      red
#     [255, 240, 150, 255],       # 5 traffic_cone    light yellow
#     [0, 175, 0, 255],           # 6 vegetation      green
#     [255, 0, 255, 255],         # 7 road            dark pink
#     [0, 175, 0, 255],           # 8 terrain         green
# ], dtype=np.float32)


# Strong Color map for visualization
COLOR_MAP = np.array([
    [255,   255,   255, 255],   # 0 undefined
    [0, 150, 245, 255],         # 1 car             blue
    [255, 192, 203, 255],       # 2 bicycle         pink
    [200, 180, 0, 255],         # 3 motorcycle      dark orange
    [255, 0, 0, 255],           # 4 pedestrian      red
    [255, 240, 150, 255],       # 5 traffic_cone    light yellow
    [0, 175, 0, 255],           # 6 vegetation      green
    [255, 0, 255, 255],         # 7 road            dark pink
    [0, 175, 0, 255],           # 8 terrain         green
    [230, 230, 250, 255],       # 9 building        white
    [0, 0, 0, 0],               # 10 Unobserved
    [128, 128, 128, 255],       # 11 Visualized voxel
], dtype=np.float32)

# Function to convert voxel data into point cloud
def _voxel2points(voxel, occ_show, voxelSize):
    occIdx = np.where(occ_show)
    points = np.concatenate((occIdx[0][:, None] * voxelSize[0],
                        occIdx[1][:, None] * voxelSize[1],
                        occIdx[2][:, None] * voxelSize[2]), axis=1)
    return points, voxel[occIdx], occIdx

# Function to generate 3D bounding box profiles for visualization
def _voxel_profile(voxel, voxel_size):
    centers = np.concatenate((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), axis=1)
    wlh = np.concatenate((np.array(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          np.array(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          np.array(voxel_size[2]).repeat(centers.shape[0])[:, None]), axis=1)
    yaw = np.full_like(centers[:, 0:1], 0)
    return np.concatenate((centers, wlh, yaw), axis=1)

# Function to compute rotation matrix along Z-axis
def _rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])

# Function to compute 3D bounding boxes
def _my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = np.expand_dims(l / 2, 1), np.expand_dims(w / 2, 1), np.expand_dims(h / 2, 1)
    x_corners = np.concatenate([-l, l, l, -l, -l, l, l, -l], axis=1)[..., None]
    y_corners = np.concatenate([w, w, -w, -w, w, w, -w, -w], axis=1)[..., None]
    z_corners = np.concatenate([h, h, h, h, -h, -h, -h, -h], axis=1)[..., None]
    corners_3d = np.concatenate([x_corners, y_corners, z_corners], axis=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def _generate_the_ego_car():
    ego_range = [-1.8, -0.8, -0.8, 1.8, 0.8, 0.8]
    ego_voxel_size = [0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    
    ego_points_label = (np.ones((ego_point_xyz.shape[0])) * 16).astype(np.uint8)
    
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    
    return ego_point_xyz

def _place_ego_car_at_position(ego_point, map_center):
    ego_point[:, 0] -= np.mean(ego_point[:, 0]) 
    ego_point[:, 1] -= np.mean(ego_point[:, 1]) 
    ego_point[:, 2] -= np.mean(ego_point[:, 2]) 
    ego_point[:, 0] += map_center[0] 
    ego_point[:, 1] += map_center[1]
    ego_point[:, 2] += map_center[2]
    return ego_point


def FillRoadInOcc(occ: np.ndarray):
    road = (occ==LABEL_ROAD)
    road_level = (np.nonzero(road)[2]).min()
    occ[:,:, road_level] = LABEL_ROAD
    return occ

def CreateOccHandle(occ: np.ndarray, free_label: int, voxelize: bool = True):
    voxel_show = occ != free_label

    colors = COLOR_MAP / 255
    points, labels, occIdx = _voxel2points(occ, voxel_show, VOXEL_SIZE)
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]
    bboxes = _voxel_profile(points, VOXEL_SIZE)
    bboxes_corners = _my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    bases_ = np.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
    edges = edges.reshape((1, 12, 2))
    edges = np.tile(edges, (bboxes_corners.shape[0], 1, 1))
    edges = edges + bases_[:, None, None]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    pcd_handle = o3d.geometry.PointCloud()
    pcd_handle.points = o3d.utility.Vector3dVector(points)
    pcd_handle.colors = o3d.utility.Vector3dVector(pcds_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])

    vis.add_geometry(pcd_handle)
    vis.add_geometry(mesh_frame)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bboxes_corners.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(edges.reshape((-1, 2)))
        line_sets.paint_uniform_color((0.5,0.5,0.5)) # Light edge: (200/255, 200/255, 200/255)
        vis.add_geometry(line_sets)

    return vis

def AddFlowToVisHandle(vis_handle, flow: np.ndarray, resolution=0.4):
    nonzero_indices = np.argwhere(np.any(flow != [0, 0, 0], axis=-1))
    nonzero_vectors = flow[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

    start_points = nonzero_indices * np.array(VOXEL_SIZE)
    end_points = start_points + nonzero_vectors * resolution

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((start_points, end_points)))
    num_lines = start_points.shape[0]
    lines = np.column_stack((np.arange(num_lines), np.arange(num_lines, 2 * num_lines)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])

    vis_handle.add_geometry(line_set)

def AddCenterEgoToVisHandle(occ, vis_handle):
    map_center = np.array(occ.shape) * VOXEL_SIZE / 2
    map_center[2] -= 1.2  # Floor correction

    ego_point = _generate_the_ego_car()
    ego_point = _place_ego_car_at_position(ego_point, map_center)

    ego_pcd = o3d.geometry.PointCloud()
    ego_pcd.points = o3d.utility.Vector3dVector(ego_point)
    vis_handle.add_geometry(ego_pcd)


def VisualizeOcc(occ: np.ndarray, free_label: int = FREE_LABEL, show_ego: bool = False):
    vis_handle = CreateOccHandle(occ, free_label)

    if show_ego:
        AddCenterEgoToVisHandle(occ, vis_handle)

    vis_handle.poll_events()
    vis_handle.update_renderer()

    return vis_handle

def VisualizeOccFlow(occ: np.ndarray, flow: np.ndarray, free_label: int = FREE_LABEL,
                     show_ego: bool = False):
    vis_handle = CreateOccHandle(occ, free_label)
    AddFlowToVisHandle(vis_handle, flow)

    if show_ego:
        AddCenterEgoToVisHandle(occ, vis_handle)

    vis_handle.poll_events()
    vis_handle.update_renderer()

    return vis_handle

def VisualizeOccFlowFile(file_path: str, free_label: int = FREE_LABEL):
    # Load File
    data = np.load(file_path, allow_pickle=True)
    voxel_label = data['occ_label']
    occ_flow = data['occ_flow_forward']

    # Visualize occ and flow
    vis_handle = CreateOccHandle(voxel_label, free_label)

    # Add flow to occ
    AddFlowToVisHandle(vis_handle, occ_flow)

    # Place ego car
    AddCenterEgoToVisHandle(voxel_label, vis_handle)

    # Show and keep window open
    while True:
        vis_handle.poll_events()
        vis_handle.update_renderer()
        if not vis_handle.poll_events():
            break

    return vis_handle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, help='Path to the NPZ file to visualize')
    args = parser.parse_args()
    
    if args.file_path is None:
        print("Error: Please provide a file path using --file_path or -f")
        exit(1)
        
    try:
        vis = VisualizeOccFlowFile(args.file_path)
        vis.destroy_window()
    except Exception as e:
        print(f"Error visualizing file: {str(e)}")
        exit(1)