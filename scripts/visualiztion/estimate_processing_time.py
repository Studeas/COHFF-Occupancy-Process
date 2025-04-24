import os
import time
import logging
from datetime import datetime
import pickle

def estimate_processing_time(input_root, output_root, verbose=True):
    """估算处理整个数据集所需的时间"""
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_root, 'logs', f'estimate_{timestamp}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting time estimation for: {input_root}")
    
    # 获取所有场景目录
    vehicle_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)) and d.startswith("2021")]
    
    if not vehicle_dirs:
        logging.error(f"No scene folders found in input directory: {input_root}")
        return
    
    # 计算总场景数和总帧数
    total_scenes = len(vehicle_dirs)
    total_frames = 0
    total_vehicles = 0
    
    for scene_dir in vehicle_dirs:
        scene_path = os.path.join(input_root, scene_dir)
        vehicle_dirs_in_scene = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
        total_vehicles += len(vehicle_dirs_in_scene)
        
        for vehicle_dir in vehicle_dirs_in_scene:
            vehicle_path = os.path.join(scene_path, vehicle_dir)
            yaml_files = [f for f in os.listdir(vehicle_path) if f.endswith('.yaml')]
            total_frames += len(yaml_files)
    
    # 计算已处理的场景和帧数
    processed_scenes = 0
    processed_frames = 0
    
    if os.path.exists(os.path.join(output_root, "scene_infos.pkl")):
        try:
            with open(os.path.join(output_root, "scene_infos.pkl"), 'rb') as f:
                existing_scene_infos = pickle.load(f)
                processed_scenes = len(existing_scene_infos)
                for scene_info in existing_scene_infos:
                    for ego_info in scene_info["egos"]:
                        processed_frames += len(ego_info["occ_in_scene_paths"])
        except Exception as e:
            logging.error(f"Error loading existing scene_infos.pkl: {str(e)}")
    
    # 计算剩余需要处理的场景和帧数
    remaining_scenes = total_scenes - processed_scenes
    remaining_frames = total_frames - processed_frames
    
    # 基于实际日志数据设置处理速度
    # 从日志中观察到处理一帧大约需要25-30秒
    avg_processing_speed = 2.4  # 帧/分钟 (60秒/25秒)
    
    # 计算预计处理时间
    estimated_total_time_hours = remaining_frames / (avg_processing_speed * 60)
    estimated_completion_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(time.time() + estimated_total_time_hours * 3600))
    
    # 输出统计信息
    logging.info(f"\n数据集统计信息:")
    logging.info(f"- 总场景数: {total_scenes}")
    logging.info(f"- 总车辆数: {total_vehicles}")
    logging.info(f"- 总帧数: {total_frames}")
    logging.info(f"- 平均每场景车辆数: {total_vehicles/total_scenes:.2f}")
    logging.info(f"- 平均每车辆帧数: {total_frames/total_vehicles:.2f}")
    
    logging.info(f"\n处理状态:")
    logging.info(f"- 已处理场景: {processed_scenes}/{total_scenes}")
    logging.info(f"- 已处理帧数: {processed_frames}/{total_frames}")
    logging.info(f"- 剩余场景: {remaining_scenes}")
    logging.info(f"- 剩余帧数: {remaining_frames}")
    
    logging.info(f"\n时间估算:")
    logging.info(f"- 估算处理速度: {avg_processing_speed:.2f} 帧/分钟")
    logging.info(f"- 预计总处理时间: {estimated_total_time_hours:.2f} 小时")
    logging.info(f"- 预计完成时间: {estimated_completion_time}")
    
    return {
        "total_scenes": total_scenes,
        "total_vehicles": total_vehicles,
        "total_frames": total_frames,
        "processed_scenes": processed_scenes,
        "processed_frames": processed_frames,
        "remaining_scenes": remaining_scenes,
        "remaining_frames": remaining_frames,
        "estimated_time_hours": estimated_total_time_hours,
        "estimated_completion_time": estimated_completion_time
    }

def main():
    # 设置输入输出路径
    input_root = "C:/semanticlidar_18_unzip/train"
    output_root = "D:/COHFF-train"
    
    # 执行时间估算
    estimate_processing_time(input_root, output_root)

if __name__ == "__main__":
    main()