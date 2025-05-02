import open3d as o3d
import numpy as np
import pickle
import os
from typing import List, Dict
import math
import pprint
import json

# 设置路径变量
# DATA_ROOT = r"D:\COHFF-train_last2"
# PKL_PATH = os.path.join(DATA_ROOT, "scene_infos.pkl")

DATA_ROOT = r"D:/"
PKL_PATH = os.path.join(DATA_ROOT, "merged_scene_info.pkl")


def load_pkl_file(pkl_path: str) -> Dict:
    """Load pkl file containing scene information"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_ego_car_mesh():
    """Create a mesh for ego car visualization"""
    # Ego car dimensions (length, width, height)
    car_size = [4.0, 1.8, 1.5]
    
    # Create a box mesh
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=car_size[0],
        height=car_size[1],
        depth=car_size[2]
    )
    
    # Paint the mesh in blue
    mesh.paint_uniform_color([0.0, 0.0, 1.0])
    
    return mesh

def create_other_car_mesh():
    """Create a mesh for other vehicles visualization"""
    # Other car dimensions
    car_size = [4.0, 1.8, 1.5]
    
    # Create a box mesh
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=car_size[0],
        height=car_size[1],
        depth=car_size[2]
    )
    
    # Paint the mesh in red
    mesh.paint_uniform_color([1.0, 0.0, 0.0])
    
    return mesh

def visualize_scene(scene_info: Dict):
    """Visualize scene information including ego and other vehicles"""
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coordinate_frame)
    
    # Process each ego vehicle in the scene
    for ego_info in scene_info['egos']:
        # Create ego car mesh
        ego_mesh = create_ego_car_mesh()
        
        # Get ego vehicle trajectory
        trajectory = []
        for frame_path in ego_info['occ_in_scene_paths']:
            # Extract position from transformation matrix
            transform = np.load(os.path.join(DATA_ROOT, frame_path))['ego_to_world_transformation']
            position = transform[:3, 3]
            trajectory.append(position)
        
        # Add trajectory line
        if len(trajectory) > 1:
            trajectory = np.array(trajectory)
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(trajectory)
            lines = [[i, i+1] for i in range(len(trajectory)-1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 1.0, 0.0])  # Green trajectory
            vis.add_geometry(line_set)
        
        # Add ego car at each position
        for pos in trajectory:
            mesh = create_ego_car_mesh()
            mesh.translate(pos)
            vis.add_geometry(mesh)
    
    # Set visualization options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])  # White background
    opt.point_size = 2.0
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def print_scene_info(scene_info: Dict):
    """打印场景信息"""
    print("\n" + "="*50)
    print(f"场景总数: {len(scene_info)}")
    print("="*50)
    
    for scene_idx, scene in enumerate(scene_info):
        print(f"\n场景 {scene_idx + 1}:")
        print(f"场景名称: {scene['scene_name']}")
        print(f"车辆数量: {len(scene['egos'])}")
        
        for ego_idx, ego in enumerate(scene['egos']):
            print(f"\n  车辆 {ego_idx + 1}:")
            print(f"  车辆名称: {ego['ego_name']}")
            print(f"  帧数: {len(ego['occ_in_scene_paths'])}")
            print("  前5帧路径:")
            for path in ego['occ_in_scene_paths'][:5]:
                print(f"    {path}")

def main():
    # 加载pkl文件
    with open(PKL_PATH, 'rb') as f:
        scene_info = pickle.load(f)
    
    # 使用pprint打印完整内容
    print("\n" + "="*50)
    print("PKL文件完整内容:")
    print("="*50 + "\n")
    pprint.pprint(scene_info, width=100, depth=None)

    # 存储为json文件
    DEBUG_PATH = os.path.join( "D:/", "scene_infos_merged.json")
    with open(DEBUG_PATH, "w") as f:
        json.dump(scene_info, f, indent=4)

if __name__ == '__main__':
    main()


ab=[1,2,3,4,5]
