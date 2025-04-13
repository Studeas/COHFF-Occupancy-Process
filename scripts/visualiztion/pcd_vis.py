import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import yaml

def plot_label_z_distribution(points, labels):
    """Plot z-direction distribution histogram for each label
    
    Args:
        points: numpy array, shape (N,3), containing xyz coordinates of point cloud
        labels: numpy array, shape (N,), containing label values for each point
    """
    # Get unique label values
    unique_labels = np.unique(labels)
    
    # Create new figure window
    plt.figure(figsize=(12, 6))
    
    # Set label color mapping
    colors = {
        0: '#FFFFFF',  # White - undefined
        1: '#0099FF',  # Blue - car
        2: '#FF9999',  # Light red - bicycle
        5: '#FFFF00',  # Yellow - traffic_cone
        6: '#00B300',  # Green - vegetation
        7: '#FF00FF',  # Purple - road
        8: '#808080',  # Gray - terrain
        9: '#CCCCCC',  # Light gray - building
        10: '#000000', # Black - unobserved
        11: '#800000', # Dark red - visualized voxel
        12: '#008080', # Cyan
        17: '#800080', # Purple
        22: '#808000'  # Olive
    }
    
    # Calculate z range
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    # Plot distribution for each label
    for label in unique_labels:
        mask = labels == label
        z_coords = points[mask, 2]
        
        # Calculate histogram data
        hist, bins = np.histogram(z_coords, bins=1000, range=(z_min, z_max)) # bins refers to the number of bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot histogram curve
        plt.plot(bin_centers, hist, label=f'Label {label}', 
                color=colors.get(label, '#808080'), linewidth=2)
    
    # Set figure properties
    plt.title('Semantic Label Distribution in Z-axis')
    plt.xlabel('Z-axis')
    plt.ylabel('Point Number')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to ensure legend is fully displayed
    plt.tight_layout()
    
    # Display figure
    plt.show()

def read_yaml_file(yaml_path):
    """Read YAML file and return annotation information"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def add_annotation_to_vis(vis, point, text, color=[1, 0, 0], size=0.5):
    """Add text annotation to visualization
    
    Args:
        vis: Visualizer object
        point: Annotation point coordinates [x, y, z]
        text: Annotation text
        color: Annotation color
        size: Annotation size
    """
    # Create sphere as annotation point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    sphere.paint_uniform_color(color)
    sphere.translate(point)
    vis.add_geometry(sphere)
    
    # Create text annotation
    text_3d = o3d.geometry.TriangleMesh.create_sphere(radius=size/2)
    text_3d.paint_uniform_color(color)
    text_3d.translate([point[0], point[1], point[2] + size*2])
    vis.add_geometry(text_3d)

def transform_coordinates(point, ego_pose):
    """Transform point from world coordinate system to ego vehicle coordinate system
    
    Args:
        point: [x, y, z] point in world coordinate system
        ego_pose: [x, y, z, roll, yaw, pitch] ego vehicle pose
    """
    # Extract ego vehicle position and orientation
    ego_pos = ego_pose[:3]
    ego_rot = ego_pose[3:]
    
    # Translate point to ego vehicle coordinate system origin
    point = np.array(point) - np.array(ego_pos)
    
    # Create rotation matrix (simplified version, only considering yaw angle)
    yaw = np.radians(ego_rot[1])
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    point = rot_matrix @ point
    
    return point

def visualize_point_cloud_with_labels(points, labels, yaml_data=None):
    """Visualize point cloud and display statistics for different labels, also show annotations from YAML
    
    Args:
        points: numpy array, shape (N,3), containing xyz coordinates of point cloud
        labels: numpy array, shape (N,), containing label values for each point
        yaml_data: Data from YAML file
    """
    # Ensure points and labels are numpy arrays
    points = np.array(points)
    labels = np.array(labels)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add ego coordinate system (red for X-axis, green for Y-axis, blue for Z-axis)
    ego_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    ego_frame.translate([0, 0, 0])  # Move coordinate system to origin
    vis.add_geometry(ego_frame)
    
    # Calculate z min and max values
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    # Add visible planes at z min and max values
    def create_highlight_plane(z_value, color):
        size = 5  # Plane size
        vertices = np.array([
            [-size, -size, z_value],
            [size, -size, z_value],
            [size, size, z_value],
            [-size, size, z_value]
        ])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    
    # Add red plane at z minimum
    min_plane = create_highlight_plane(z_min, [1, 0, 0])  # Red
    vis.add_geometry(min_plane)
    
    # Add green plane at z maximum
    max_plane = create_highlight_plane(z_max, [0, 1, 0])  # Green
    vis.add_geometry(max_plane)
    
    # Create point cloud with different colors for each label
    unique_labels = np.unique(labels)  # Get unique label values
    colors = {
        0: [1, 1, 1],      # White - undefined
        1: [0, 0.6, 1],    # Blue - car
        2: [1, 0.8, 0.8],  # Light red - bicycle
        5: [1, 1, 0],      # Yellow - traffic_cone
        6: [0, 0.7, 0],    # Green - vegetation
        7: [1, 0, 1],      # Purple - road
        8: [0.5, 0.5, 0.5],# Gray - terrain
        9: [0.8, 0.8, 0.8],# Light gray - building
        10: [0, 0, 0],     # Black - unobserved
        11: [0.5, 0, 0],   # Dark red - visualized voxel
        12: [0, 0.5, 0.5], # Cyan
        17: [0.5, 0, 0.5], # Purple
        22: [0.5, 0.5, 0]  # Olive
    }
    
    # Calculate statistics for each label
    print("\nLabel Statistics:")
    print("-" * 50)
    print(f"{'Label':<6} {'Points':<10} {'Mean Z':<10} {'Min Z':<10} {'Max Z':<10}")
    print("-" * 50)
    
    # Use set to ensure each label is printed only once
    printed_labels = set()
    
    for label in unique_labels:
        if label not in printed_labels:  # Only print unprinted labels
            mask = labels == label
            label_points = points[mask]
            z_coords = label_points[:, 2]
            
            print(f"{label:<6} {np.sum(mask):<10} {np.mean(z_coords):<10.3f} "
                  f"{np.min(z_coords):<10.3f} {np.max(z_coords):<10.3f}")
            
            # Create point cloud for this label
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(label_points)
            pcd.paint_uniform_color(colors.get(label, [0.5, 0.5, 0.5]))  # Default gray
            vis.add_geometry(pcd)
            
            printed_labels.add(label)  # Mark label as printed
    
    print("-" * 50)
    
    # Add annotations from YAML
    if yaml_data:
        print("\nYAML Annotation Information:")
        print("-" * 50)
        
        # Get ego vehicle pose
        ego_pose = None
        if 'lidar_pose' in yaml_data:
            ego_pose = yaml_data['lidar_pose']
            ego_pos = ego_pose[:3]
            print(f"Ego Vehicle world position: {ego_pos}")
            
            # Transform ego vehicle position to point cloud coordinate system
            ego_pos_local = transform_coordinates(ego_pos, ego_pose)
            add_annotation_to_vis(vis, ego_pos_local, "Ego Vehicle", [1, 1, 0])
            print(f"Ego Vehicle local position: {ego_pos_local}")
        
        # Add other vehicle annotations
        if 'vehicles' in yaml_data:
            for vehicle_id, vehicle_info in yaml_data['vehicles'].items():
                if 'location' in vehicle_info:
                    world_pos = vehicle_info['location']
                    print(f"Vehicle {vehicle_id} world position: {world_pos}")
                    
                    # Transform vehicle position to point cloud coordinate system
                    local_pos = transform_coordinates(world_pos, ego_pose)
                    add_annotation_to_vis(vis, local_pos, f"Vehicle {vehicle_id}", [1, 0, 0])
                    print(f"Vehicle {vehicle_id} local position: {local_pos}")
        
        # Add camera position annotations
        for i in range(4):
            camera_key = f'camera{i}'
            if camera_key in yaml_data:
                world_pos = yaml_data[camera_key]['cords'][:3]
                print(f"Camera {i} world position: {world_pos}")
                
                # Transform camera position to point cloud coordinate system
                local_pos = transform_coordinates(world_pos, ego_pose)
                add_annotation_to_vis(vis, local_pos, f"Camera {i}", [0, 1, 1])
                print(f"Camera {i} local position: {local_pos}")
        
        print("-" * 50)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def read_pcd_file(pcd_path):
    """Read PCD file and return point cloud data and labels"""
    points = []
    labels = []
    with open(pcd_path, 'r') as f:
        # Skip header information
        for line in f:
            if line.strip() == 'DATA ascii':
                break
        # Read point cloud data
        for line in f:
            if len(line.split()) >= 4:
                x, y, z, label = map(float, line.split())
                points.append([x, y, z])
                labels.append(int(label))
    return np.array(points), np.array(labels)

if __name__ == "__main__":
    # Set file paths
    pcd_path = r"/Users/xiaokangsun/Documents/data_example/validate/2021_08_18_19_48_05/1045/000070_semantic_occluded.pcd"
    yaml_path = r"/Users/xiaokangsun/Documents/data_example/validate/2021_08_18_19_48_05/1045/000070.yaml"
    
    # Read point cloud data and labels
    points, labels = read_pcd_file(pcd_path)
    
    # Read YAML data
    yaml_data = read_yaml_file(yaml_path)
    
    # Visualize point cloud and annotations
    visualize_point_cloud_with_labels(points, labels, yaml_data)

