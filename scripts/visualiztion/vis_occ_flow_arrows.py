import numpy as np
import open3d as o3d
import os

def create_simple_arrow(scale=1.0, color=[1, 0, 0]):
    """Create a simple arrow"""
    # Create cylinder (arrow body)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.6)
    # Create cone (arrow head)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.1, height=0.1)
    # Move cone to top of cylinder
    cone.translate([0, 0, 0.6])
    # Combine geometries
    arrow = o3d.geometry.TriangleMesh()
    arrow += cylinder
    arrow += cone
    # Scale and color
    arrow.scale(scale, center=arrow.get_center())
    arrow.paint_uniform_color(color)
    return arrow

def visualize_occ_flow_arrows(npz_path):
    """Visualize occ_flow arrows"""
    # Load data
    data = np.load(npz_path)
    occ_flow = data['occ_flow']
    
    # Calculate flow magnitude
    flow_magnitude = np.sqrt(np.sum(occ_flow**2, axis=3))
    
    # Print statistics
    total_points = flow_magnitude.size
    non_zero_points = np.sum(flow_magnitude > 0)
    zero_points = total_points - non_zero_points
    
    print(f"\nFlow Statistics:")
    print(f"Total points: {total_points}")
    print(f"Non-zero points: {non_zero_points}")
    print(f"Zero points: {zero_points}")
    print(f"Non-zero point ratio: {non_zero_points/total_points*100:.2f}%")
    
    # Print magnitude distribution
    non_zero_magnitudes = flow_magnitude[flow_magnitude > 0]
    print(f"\nNon-zero flow magnitude distribution:")
    print(f"Minimum: {np.min(non_zero_magnitudes):.4f}")
    print(f"Maximum: {np.max(non_zero_magnitudes):.4f}")
    print(f"Mean: {np.mean(non_zero_magnitudes):.4f}")
    print(f"Median: {np.median(non_zero_magnitudes):.4f}")
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Set magnitude threshold to only show larger flows
    magnitude_threshold = np.percentile(flow_magnitude[flow_magnitude > 0], 10)  # Lower to 50th percentile
    
    # Get all non-zero flow points
    x, y, z = np.where(flow_magnitude > magnitude_threshold)
    
    # Sample every N points for performance
    N = 20  # Increase sampling interval to reduce number of arrows
    
    # Pre-calculate all arrow geometries
    arrows = []
    for i in range(0, len(x), N):
        # Get flow vector and magnitude
        flow = occ_flow[x[i], y[i], z[i]]
        magnitude = flow_magnitude[x[i], y[i], z[i]]
        
        # Create arrow
        arrow = create_simple_arrow(scale=magnitude)
        
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
        color = (direction + 1) / 2  # Map [-1,1] to [0,1]
        arrow.paint_uniform_color(color)
        
        arrows.append(arrow)
    
    # Add geometries in batch
    for arrow in arrows:
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
        
    visualize_occ_flow_arrows(npz_path)
    print("\nVisualization completed!")

if __name__ == "__main__":
    main() 