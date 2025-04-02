import numpy as np
import os

def view_npz_content(npz_path):
    """
    View npz file content and print basic information
    """
    try:
        # Load npz file
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"\n{'='*50}")
        print(f"File path: {npz_path}")
        print(f"{'='*50}")
        
        # Display array names and basic information
        print("\nArrays contained:")
        print('-'*30)
        for key in data.files:
            array = data[key]
            print(f"\nArray name: {key}")
            print(f"Shape: {array.shape}")
            print(f"Type: {array.dtype}")
            
            # Process differently based on array type
            if key == 'annotations':
                print("\nAnnotation content:")
                for i, ann in enumerate(array):
                    print(f"\nAnnotation {i+1}:")
                    # 直接处理字典类型
                    for k, v in ann.items():
                        if isinstance(v, np.ndarray):
                            print(f"  {k}: shape={v.shape}, range=[{v.min():.2f}, {v.max():.2f}]")
                        else:
                            print(f"  {k}: {v}")
            else:
                # For other arrays, print value ranges
                if isinstance(array, np.ndarray):
                    if array.dtype.kind in 'ui':  # Integer type
                        print(f"Value range: [{array.min()}, {array.max()}]")
                    else:  # Float type
                        print(f"Value range: [{array.min():.2f}, {array.max():.2f}]")
            
            print('-'*30)
            
    except Exception as e:
        print(f"Error: Cannot read file {npz_path}")
        print(f"Error message: {str(e)}")

def main():
    # Set npz file path
    npz_path = r"C:\Users\TUF\Desktop\backup\data_example\COHFF-val2\scene_2021_08_18_19_48_05\Ego1045\000068.npz"
    view_npz_content(npz_path)

if __name__ == "__main__":
    main()