import numpy as np
import os
import pprint

def view_npz_content(npz_path):
    """
    View npz file content and print complete structure
    """
    try:
        # Load npz file
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"\n{'='*50}")
        print(f"File path: {npz_path}")
        print(f"{'='*50}")
        
        # 创建完整的数据结构字典
        content_dict = {}
        for key in data.files:
            array = data[key]
            print(f"\n处理数组: {key}")
            print(f"类型: {type(array)}")
            
            if key == 'annotations':
                # 处理annotations数组
                annotations_list = []
                for i, ann in enumerate(array):
                    print(f"\n处理标注 {i+1}:")
                    ann_dict = {}
                    for k, v in ann.items():
                        print(f"  {k}: {type(v)}")
                        if isinstance(v, np.ndarray):
                            ann_dict[k] = {
                                'shape': v.shape,
                                'dtype': str(v.dtype),
                                'min': float(v.min()),
                                'max': float(v.max())
                            }
                        else:
                            ann_dict[k] = v
                    annotations_list.append(ann_dict)
                content_dict[key] = annotations_list
            elif key == 'cameras':
                # 处理cameras数组
                cameras_list = []
                for i, cam in enumerate(array):
                    print(f"\n处理相机 {i+1}:")
                    cam_dict = {}
                    for k, v in cam.items():
                        print(f"  {k}: {type(v)}")
                        if isinstance(v, np.ndarray):
                            cam_dict[k] = {
                                'shape': v.shape,
                                'dtype': str(v.dtype),
                                'content': v.tolist()
                            }
                        else:
                            cam_dict[k] = v
                    cameras_list.append(cam_dict)
                content_dict[key] = cameras_list
            else:
                # 处理其他数组
                if isinstance(array, np.ndarray):
                    if array.dtype.kind in 'U':  # 字符串类型
                        content_dict[key] = array.item()
                    else:
                        try:
                            content_dict[key] = {
                                'shape': array.shape,
                                'dtype': str(array.dtype),
                                'min': float(array.min()),
                                'max': float(array.max())
                            }
                        except Exception as e:
                            print(f"处理数组 {key} 时出错: {str(e)}")
                            content_dict[key] = {
                                'shape': array.shape,
                                'dtype': str(array.dtype),
                                'content': array.tolist()
                            }
                else:
                    # 处理非numpy数组类型
                    content_dict[key] = array
        
        # 使用pprint打印完整结构
        print("\n完整数据结构:")
        pprint.pprint(content_dict, width=100, depth=None)
            
    except Exception as e:
        print(f"Error: Cannot read file {npz_path}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Stack trace:\n{traceback.format_exc()}")

def main():
    # Set npz file path
    npz_path = r"C:\Users\TUF\Desktop\backup\data_example\COHFF-val4\2021_08_18_19_48_05\1045\000068.npz"
    view_npz_content(npz_path)

if __name__ == "__main__":
    main()