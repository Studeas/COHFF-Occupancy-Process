# OPV2V Data Processing Guide

## Overview
`uni_operation.py` processes OPV2V dataset into CoHFF format, handling point clouds, camera images, and sensor data.

## Workflow
1. **Data Organization**
   - Input: Scene folders with vehicle data
   - Output: CoHFF format data
   - Structure: Scenes → Vehicles → Frames

2. **Frame Processing**
   - Point cloud: Read, crop, transform, voxelize
   - Labels: Remap semantic labels
   - Flow: Calculate between consecutive frames
   - Sensors: Process FOV masks and calibration
   - Annotations: Process vehicle information

3. **Output Generation**
   - NPZ files: Processed data
   - Camera images: Copied and organized
   - Scene info: Pickle file with metadata

## Usage
```bash
python uni_operation.py --input_root <input_path> --output_root <output_path> [--verbose]
```

### Directory Structure
```
input_root/
├── 2021_08_18_19_48_05/
│   ├── 1045/
│   │   ├── 000068_semantic_occluded.pcd
│   │   ├── 000068_camera0.png
│   │   ├── 000068.yaml
│   └── ...
└── ...

output_root/
├── scene_2021_08_18_19_48_05/
│   ├── Ego1045/
│   │   ├── 000068.npz
│   │   └── 000068_camera0.png
├── logs/
└── scene_infos.pkl
```

### Key Outputs
1. **NPZ Files**
   - `occ_label`: Voxel grid
   - `occ_flow`: Flow vectors
   - `occ_mask_*`: FOV masks
   - Vehicle pose and annotations
   - Camera parameters

2. **Scene Info**
   - `scene_infos.pkl`: Dataset metadata
   - File paths and organization

## Notes
- Required: PCD, YAML, PNG camera images
- Check logs for processing status
- Large datasets may take significant time
