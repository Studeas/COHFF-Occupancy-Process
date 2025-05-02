import json
import pickle
import os

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_scene_infos(json1, json2):
    """合并两个场景信息列表"""
    return json1 + json2

def save_to_pkl(data, pkl_path):
    """将数据保存为PKL文件"""
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    # 定义JSON文件路径
    json_file1 = 'D:/scene_infos.json'
    json_file2 = 'D:/scene_infos2.json'
    
    # 加载JSON文件
    scene_info1 = load_json_file(json_file1)
    scene_info2 = load_json_file(json_file2)
    
    # 合并场景信息
    merged_scene_info = merge_scene_infos(scene_info1, scene_info2)
    
    # 定义输出PKL文件路径
    output_pkl_path = 'D:/merged_scene_info.pkl'
    
    # 保存合并后的数据为PKL文件
    save_to_pkl(merged_scene_info, output_pkl_path)
    print(f"合并后的数据已保存到 {output_pkl_path}")

if __name__ == '__main__':
    main()