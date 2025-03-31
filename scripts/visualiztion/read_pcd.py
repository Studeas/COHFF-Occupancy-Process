import numpy as np
import open3d as o3d

def get_pcd_size(pcd_path):
    """
    Read pcd file and return point cloud size and spatial range
    
    Args:
        pcd_path: Path to the pcd file
        
    Returns:
        points_num: Number of points in the point cloud
        dimensions: Range of point cloud in x, y, z dimensions
        min_bound: Minimum boundary values
        max_bound: Maximum boundary values
    """
    # Read pcd file
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Get point cloud data
    points = np.asarray(pcd.points)
    
    # Calculate number of points
    points_num = len(points)
    
    # Calculate min and max values for each dimension
    if points_num > 0:
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        dimensions = max_bound - min_bound
    else:
        min_bound = np.zeros(3)
        max_bound = np.zeros(3)
        dimensions = np.zeros(3)
    
    return points_num, dimensions, min_bound, max_bound

def points_to_voxel_grid(pcd_path, grid_size=(200, 200, 16)):
    """
    Convert point cloud to fixed-size voxel grid
    
    Args:
        pcd_path: Path to the pcd file
        grid_size: Size of the voxel grid, default is (200, 200, 16)
        
    Returns:
        voxel_grid: 3D numpy array with shape grid_size
    """
    # Read point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    # Return empty grid if point cloud is empty
    if len(points) == 0:
        return np.zeros(grid_size)
    
    # Calculate point cloud boundaries
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # Initialize voxel grid
    voxel_grid = np.zeros(grid_size)
    
    # Calculate step size for each dimension
    steps = (max_bound - min_bound) / np.array([grid_size[0], grid_size[1], grid_size[2]])
    
    # Map point cloud points to voxel grid
    for point in points:
        # Calculate grid indices for point
        idx = np.floor((point - min_bound) / steps).astype(int)
        
        # Ensure indices are within valid range
        idx = np.clip(idx, [0, 0, 0], [grid_size[0]-1, grid_size[1]-1, grid_size[2]-1])
        
        # Mark 1 at corresponding voxel position
        voxel_grid[idx[0], idx[1], idx[2]] = 1
    
    return voxel_grid

if __name__ == "__main__":
    pcd_path = r"E:\semantic_data\semanticlidar_18_unzip\train\2021_08_16_22_26_54\641\000069_semantic_occluded.pcd"
    
    # Get basic point cloud information
    points_num, dimensions, min_bound, max_bound = get_pcd_size(pcd_path)
    print(f"Number of points in point cloud: {points_num}")
    print(f"Point cloud spatial range (x × y × z): {dimensions[0]:.2f}m × {dimensions[1]:.2f}m × {dimensions[2]:.2f}m")
    print(f"Point cloud boundary range:")
    print(f"X-axis range: {min_bound[0]:.2f}m to {max_bound[0]:.2f}m")
    print(f"Y-axis range: {min_bound[1]:.2f}m to {max_bound[1]:.2f}m")
    print(f"Z-axis range: {min_bound[2]:.2f}m to {max_bound[2]:.2f}m")
    
    # Convert to voxel grid
    voxel_grid = points_to_voxel_grid(pcd_path)
    print(f"\nVoxel grid shape: {voxel_grid.shape}")
    print(f"Number of non-zero elements in voxel grid: {np.count_nonzero(voxel_grid)}")
    print(f"Occupancy rate: {(np.count_nonzero(voxel_grid) / voxel_grid.size * 100):.2f}%")

