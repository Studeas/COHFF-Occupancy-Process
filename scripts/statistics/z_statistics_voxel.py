import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_voxel_file(file_path):
    """Read NPZ file and return voxelized label grid"""
    data = np.load(file_path)
    return data['occ_label']    

def plot_z_distribution(voxel_label):
    """Plot z-axis distribution histogram"""
    # Get voxel grid shape
    H, W, D, _ = voxel_label.shape
    
    # Count non-zero points for each z-layer
    z_counts = []
    z_values = []
    
    # z-axis actual range is [-3.0, 3.4], total 16 layers
    z_range = np.linspace(-3.0, 3.4, D)
    voxel_z = (3.4 - (-3.0)) / D  # Height of each voxel
    
    print("\nZ-axis Statistics:")
    print("-" * 50)
    print(f"{'Z Range':<20} {'Points':<10} {'Percentage':<15}")
    print("-" * 50)
    
    total_points = np.sum(voxel_label > 0)
    
    for z_idx in range(D):
        # Count non-zero points in current z-layer
        count = np.sum(voxel_label[:, :, z_idx, 0] > 0)
        z_counts.append(count)
        
        # Calculate actual z-value range for current z-layer
        z_min = z_range[z_idx] - voxel_z/2
        z_max = z_range[z_idx] + voxel_z/2
        z_values.append(z_range[z_idx])
        
        # Print statistics
        percentage = (count / total_points * 100) if total_points > 0 else 0
        print(f"[{z_min:6.2f}, {z_max:6.2f}] {count:<10d} {percentage:6.2f}%")
    
    print("-" * 50)
    print(f"Total Points: {total_points}")
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(z_values, z_counts, width=voxel_z*0.8, align='center', alpha=0.8)
    plt.xlabel('Z-axis Position (m)')
    plt.ylabel('Point Count')
    plt.title('Point Distribution Across Z-layers in Voxel Grid')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(z_counts):
        plt.text(z_values[i], count, str(count), ha='center', va='bottom')

def main(file_path):
    """Main function"""
    voxel_label = read_voxel_file(file_path)
    plot_z_distribution(voxel_label)
    plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045_cropped_voxel\000068_semantic_occluded_voxel.npz"
    main(file_path)
