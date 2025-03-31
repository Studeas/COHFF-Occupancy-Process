import open3d as o3d
import numpy as np
import argparse

# Label mapping dictionary
opv2v_label_mapping_dict = {
    0: 0,  # unlabeled
    1: 1,  # Building
    2: 2,  # Fence
    3: 0,  # unlabeled
    4: 0,  # unlabeled
    5: 4,  # Pole
    6: 5,  # Road
    7: 5,  # Road
    8: 6,  # SideWalk
    9: 7,  # Vegetation
    10: 8,  # Vehicles
    11: 9,  # Wall
    12: 11, # TrafficSign
    13: 0,  # unlabeled
    14: 3,  # Ground
    15: 12, # Bridge
    16: 0,  # unlabeled
    17: 10, # GuardRail
    18: 4,  # Pole
    19: 0,  # unlabeled
    20: 8,  # Vehicles
    21: 0,  # unlabeled
    22: 3   # Terrain
}

# Color mapping
COLOR_MAP = np.array([
    [255, 255, 255, 255],  # 0 unlabeled (white)
    [255, 0, 0, 255],      # 1 Building (red)
    [0, 255, 0, 255],      # 2 Fence (green)
    [0, 0, 255, 255],      # 3 Ground/Terrain (blue)
    [255, 255, 0, 255],    # 4 Pole (yellow)
    [255, 0, 255, 255],    # 5 Road (purple)
    [0, 255, 255, 255],    # 6 SideWalk (cyan)
    [128, 255, 0, 255],    # 7 Vegetation (yellow-green)
    [255, 128, 0, 255],    # 8 Vehicles (orange)
    [0, 128, 255, 255],    # 9 Wall (light blue)
    [255, 0, 128, 255],    # 10 GuardRail (pink)
    [128, 0, 255, 255],    # 11 TrafficSign (purple)
    [0, 255, 128, 255],    # 12 Bridge (cyan-green)
], dtype=np.float32)

def visualize_voxel_grid(voxel_grid):
    """Visualize voxel grid"""
    # Remove the last dimension
    voxel_grid = voxel_grid.squeeze()
    
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Get coordinates of non-zero voxels
    x, y, z = np.nonzero(voxel_grid)
    points = np.column_stack((x, y, z))
    
    # Set point cloud coordinates
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Set point cloud colors (different colors based on label values)
    colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        label = voxel_grid[x[i], y[i], z[i]]
        # Use label mapping
        mapped_label = opv2v_label_mapping_dict.get(label, 0)  # Default to 0 if label not in mapping
        # Set color
        colors[i] = COLOR_MAP[mapped_label][:3] / 255.0  # Normalize to [0,1] range
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add point cloud to visualizer
    vis.add_geometry(pcd)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main(file_path):
    # Load NPZ file
    data = np.load(file_path)
    voxel_label = data['occ_label']
    
    print(f"Voxel grid shape: {voxel_label.shape}")
    print(f"Unique label values: {np.unique(voxel_label)}")
    
    # Visualize
    visualize_voxel_grid(voxel_label)

if __name__ == '__main__':
    file_path = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045_cropped_voxel\000068_semantic_occluded_voxel.npz"
    # file_path = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045_cropped_voxel\000070_semantic_occluded_voxel.npz"
    
    main(file_path)