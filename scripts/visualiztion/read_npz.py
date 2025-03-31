import numpy as np
import os

def view_npz_content(npz_path):
    """
    View npz file content and save in readable format
    """
    try:
        # Load npz file
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"\n{'='*50}")
        print(f"File path: {npz_path}")
        print(f"{'='*50}")
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(npz_path), 'npz_visualization')
        output_dir = output_dir.replace('D:/CoHFF-val/scene_0001/Ego2149/', 'C:/Users/TUF/Desktop/opv2v_process/')
        os.makedirs(output_dir, exist_ok=True)
        
        # Display array names and basic information
        print("\nArrays contained:")
        print('-'*30)
        for key in data.files:
            array = data[key]
            print(f"\nArray name: {key}")
            print(f"Shape: {array.shape}")
            print(f"Type: {array.dtype}")
            
            # Process differently based on array type and dimensions
            if key == 'annotations':
                print("\nAnnotation content:")
                for i, ann in enumerate(array):
                    print(f"\nAnnotation {i+1}:")
                    for k, v in ann.item().items():
                        print(f"  {k}: {v}")
            else:
                # For 3D or 4D arrays, save each slice
                if len(array.shape) >= 3:
                    # Create output directory for this array
                    array_dir = os.path.join(output_dir, key)
                    os.makedirs(array_dir, exist_ok=True)
                    
                    # Save statistics
                    stats_file = os.path.join(array_dir, 'statistics.txt')
                    with open(stats_file, 'w') as f:
                        f.write(f"Array name: {key}\n")
                        f.write(f"Shape: {array.shape}\n")
                        f.write(f"Type: {array.dtype}\n")
                        f.write(f"Minimum value: {array.min()}\n")
                        f.write(f"Maximum value: {array.max()}\n")
                        f.write(f"Mean value: {array.mean()}\n")
                    
                    # Save each slice
                    if len(array.shape) == 3:  # 3D array
                        for i in range(array.shape[0]):
                            slice_file = os.path.join(array_dir, f'slice_{i:03d}.txt')
                            np.savetxt(slice_file, array[i], fmt='%d' if array.dtype.kind in 'ui' else '%.6f')
                    elif len(array.shape) == 4:  # 4D array
                        for i in range(array.shape[0]):
                            for j in range(array.shape[1]):
                                slice_file = os.path.join(array_dir, f'slice_{i:03d}_{j:03d}.txt')
                                np.savetxt(slice_file, array[i,j], fmt='%.6f')
                    
                    print(f"Saved slices to directory: {array_dir}")
                else:
                    # For 1D or 2D arrays, save directly
                    output_file = os.path.join(output_dir, f'{key}.txt')
                    output_file = output_file.replace('D:/CoHFF-val/scene_0001/Ego2149/', 'C:/Users/TUF/Desktop/opv2v_process/')
                    np.savetxt(output_file, array, fmt='%d' if array.dtype.kind in 'ui' else '%.6f')
                    print(f"Saved to file: {output_file}")
            
            print('-'*30)
            
    except Exception as e:
        print(f"Error: Cannot read file {npz_path}")
        print(f"Error message: {str(e)}")

def main():
    # Set npz file path
    npz_path = r"C:/Users/TUF/Desktop/opv2v_process/CoHFF-val/scene_0001/Ego1045/0000000001.npz"
    view_npz_content(npz_path)

if __name__ == "__main__":
    main()