import numpy as np
import os

def npz_to_txt(npz_path):
    """Convert npz file to txt format
    Args:
        npz_path: Path to the npz file
    """
    # Load npz file
    data = np.load(npz_path)
    
    # Create output directory
    output_dir = os.path.dirname(npz_path)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    
    # Print npz file structure
    print(f"\nNPZ {npz_path} structure:")
    print("-" * 50)
    for key in data.files:
        array = data[key]
        print(f"key: {key}")
        print(f"shape: {array.shape}")
        print(f"type: {array.dtype}")
        print("-" * 30)

    # Create txt file for each array
    for key in data.files:
        array = data[key]
        output_file = os.path.join(output_dir, f"{base_name}_{key}.txt")
        
        # Save array information to txt file
        with open(output_file, 'w') as f:
            # Write array metadata
            f.write(f"array name: {key}\n")
            f.write(f"shape: {array.shape}\n")
            f.write(f"type: {array.dtype}\n")
            f.write("-" * 50 + "\n")
            
            # Choose saving method based on array dimensions
            if len(array.shape) <= 2:
                # Save 1D or 2D arrays directly
                np.savetxt(f, array, fmt='%d' if array.dtype.kind in 'ui' else '%.6f')
            else:
                # Save 3D or higher dimensional arrays layer by layer
                for i in range(array.shape[0]):
                    f.write(f"\nlayer {i}:\n")
                    if len(array.shape) == 3:
                        np.savetxt(f, array[i], fmt='%d' if array.dtype.kind in 'ui' else '%.6f')
                    elif len(array.shape) == 4:
                        for j in range(array.shape[1]):
                            f.write(f"\nsublayer {j}:\n")
                            np.savetxt(f, array[i,j], fmt='%d' if array.dtype.kind in 'ui' else '%.6f')
        
        print(f"saved to file: {output_file}")

def main():
    # Example usage
    npz_path = r"C:\Users\TUF\Desktop\opv2v_process\single_data_example\2021_08_18_19_48_05\1045_cropped_voxel\000068_semantic_occluded_voxel.npz"
    
    if not os.path.exists(npz_path):
        print(f"error: file not found {npz_path}")
        return
        
    npz_to_txt(npz_path)

if __name__ == "__main__":
    main() 