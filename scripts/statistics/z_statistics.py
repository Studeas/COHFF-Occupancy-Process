import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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

def plot_all_pcd_z_distribution(folder_path):
    """Calculate and plot z-direction distribution of all semantic point clouds in the folder
    
    Args:
        folder_path: Folder path containing PCD files
    """
    # Get all semantic point cloud files
    pcd_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_semantic_occluded.pcd'):
                pcd_files.append(os.path.join(root, file))
    
    if not pcd_files:
        print(f"No semantic point cloud files found in {folder_path}")
        return
    
    print(f"Found {len(pcd_files)} semantic point cloud files")
    
    # Create new figure window
    plt.figure(figsize=(15, 8))
    
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
    
    # For storing z-value ranges of all files
    all_z_min = float('inf')
    all_z_max = float('-inf')
    
    # For storing statistics of each label
    label_stats = {}
    
    # First traverse all files to get z-value range
    print("Calculating z-value range...")
    for pcd_file in tqdm(pcd_files):
        points, labels = read_pcd_file(pcd_file)
        all_z_min = min(all_z_min, np.min(points[:, 2]))
        all_z_max = max(all_z_max, np.max(points[:, 2]))
    
    print(f"Z-value range: {all_z_min:.3f} to {all_z_max:.3f}")
    
    # Create histogram data for each label
    print("Calculating label distribution...")
    for label in colors.keys():
        label_stats[label] = np.zeros(1000)  # Use 1000 bins
    
    # Traverse all files and accumulate histogram data
    for pcd_file in tqdm(pcd_files):
        points, labels = read_pcd_file(pcd_file)
        for label in colors.keys():
            mask = labels == label
            if np.any(mask):
                z_coords = points[mask, 2]
                hist, _ = np.histogram(z_coords, bins=1000, range=(all_z_min, all_z_max))
                label_stats[label] += hist
    
    # Calculate bin centers
    bin_edges = np.linspace(all_z_min, all_z_max, 1001)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot distribution curve for each label
    print("Plotting distribution...")
    for label, hist in label_stats.items():
        if np.any(hist > 0):  # Only plot labels with points
            plt.plot(bin_centers, hist, label=f'Label {label}', 
                    color=colors.get(label, '#808080'), linewidth=2)
    
    # Set figure properties
    plt.title('All Semantic Point Clouds Z-axis Distribution')
    plt.xlabel('Z-axis')
    plt.ylabel('Total Point Number')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to ensure legend is fully displayed
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(folder_path, 'z_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    
    # Display figure
    plt.show()
    
    # Print statistics
    print("\nLabel Statistics:")
    print("-" * 50)
    print(f"{'Label':<6} {'Total Points':<12} {'Mean Z':<10} {'Min Z':<10} {'Max Z':<10}")
    print("-" * 50)
    
    for label, hist in label_stats.items():
        if np.any(hist > 0):
            total_points = np.sum(hist)
            z_values = bin_centers[hist > 0]
            weights = hist[hist > 0]
            mean_z = np.average(z_values, weights=weights)
            min_z = bin_centers[np.argmax(hist > 0)]
            max_z = bin_centers[np.argmax(hist[::-1] > 0)]
            
            print(f"{label:<6} {total_points:<12} {mean_z:<10.3f} "
                  f"{min_z:<10.3f} {max_z:<10.3f}")
    
    print("-" * 50)

if __name__ == "__main__":
    # Set folder path to analyze
    folder_path0 = r"C:\Users\TUF\Desktop\backup\data_example\validate\2021_08_18_19_48_05\1045"
    
    # Analyze and plot distribution
    plot_all_pcd_z_distribution(folder_path0)

    # folder_path1 = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045"
    # plot_all_pcd_z_distribution(folder_path1)
