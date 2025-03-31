import numpy as np
import open3d as o3d
import os

def create_arrow_mesh(scale=1.0, color=[1, 0, 0]):
    """Create an arrow mesh"""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.1,
        cone_radius=0.2,
        cylinder_height=0.8,
        cone_height=0.2,
        resolution=20,
        cylinder_split=4,
        cone_split=1
    )
    arrow.scale(scale, center=arrow.get_center())
    arrow.paint_uniform_color(color)
    return arrow

def get_flow_color(flow):
    """Return color based on flow direction"""
    # Normalize flow direction
    direction = flow / (np.linalg.norm(flow) + 1e-6)
    
    # Use direction vector as RGB color
    # Map [-1,1] range to [0,1] range
    color = (direction + 1) / 2
    return color

def visualize_occ_flow_3d(npz_path):
    """Visualize occ_flow data using Open3D"""
    # Load data
    data = np.load(npz_path)
    occ_flow = data['occ_flow']
    
    # Print basic information
    print("\n=== Occ Flow Basic Information ===")
    print(f"Shape: {occ_flow.shape}")
    print(f"Min value: {np.min(occ_flow):.4f}")
    print(f"Max value: {np.max(occ_flow):.4f}")
    
    # Calculate flow magnitude
    flow_magnitude = np.sqrt(np.sum(occ_flow**2, axis=3))
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Add voxel point cloud
    # Show all points
    x, y, z = np.where(flow_magnitude > 0)  # Only show points with flow
    points = np.column_stack((x, y, z))
    
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Set point cloud colors (based on flow magnitude)
    colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        magnitude = flow_magnitude[x[i], y[i], z[i]]
        # Use heatmap color mapping
        colors[i] = [1, 0, 0] if magnitude > np.max(flow_magnitude) * 0.8 else [0, 1, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add point cloud to visualizer
    vis.add_geometry(pcd)
    
    # Add flow arrows
    # Sample every N points for performance
    N = 2  # Sampling interval
    for i in range(0, len(points), N):
        flow = occ_flow[x[i], y[i], z[i]]
        magnitude = flow_magnitude[x[i], y[i], z[i]]
        
        # Create arrow
        arrow = create_arrow_mesh(scale=magnitude)
        
        # Calculate arrow rotation
        direction = flow / (magnitude + 1e-6)
        rotation = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        if np.any(direction != [0, 0, 1]):
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.cross([0, 0, 1], direction)
            )
        
        # Set arrow position and rotation
        arrow.translate([x[i], y[i], z[i]])
        arrow.rotate(rotation, center=arrow.get_center())
        
        # Set arrow color (based on flow direction)
        arrow_color = get_flow_color(flow)
        arrow.paint_uniform_color(arrow_color)
        
        # Add arrow to visualizer
        vis.add_geometry(arrow)
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    # Example usage
    npz_path = r"C:\Users\TUF\Desktop\opv2v_process\data_example\2021_08_18_19_48_05\1045\occ_flow.npz"
    
    if not os.path.exists(npz_path):
        print(f"Error: File not found {npz_path}")
        return
        
    visualize_occ_flow_3d(npz_path)
    print("\nVisualization completed!")

if __name__ == "__main__":
    main() 