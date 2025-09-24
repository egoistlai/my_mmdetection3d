# tools/data_converter/drad_converter.py (最终完整版)
import os
import numpy as np
import pickle
from os.path import join
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

# ==============================================================================
# --- HELPER FUNCTIONS FOR GROUND TRUTH DATABASE CREATION (LiDAR COORDINATES) ---
# ==============================================================================

def rotz_to_mat(rotation_z):
    """ 从绕Z轴的旋转（LiDAR yaw）创建3x3旋转矩阵 """
    cos_rz = np.cos(rotation_z)
    sin_rz = np.sin(rotation_z)
    rot_mat = np.array([
        [cos_rz, -sin_rz, 0],
        [sin_rz,  cos_rz, 0],
        [0, 0, 1]
    ])
    return rot_mat

def points_in_lidar_rbox(points, rbox):
    """ 检查哪些点在LiDAR坐标系下的旋转3D框内 """
    center = rbox[:3]
    dimensions = rbox[3:6] # l, w, h
    rotation_z = rbox[6]
    
    rot_mat = rotz_to_mat(rotation_z)
    inv_rot_mat = rot_mat.T

    translated_points = points[:, :3] - center
    rotated_points = translated_points @ inv_rot_mat
    
    half_dims = dimensions / 2.0
    is_inside = np.all(np.abs(rotated_points) <= half_dims, axis=1)
    
    return is_inside

def lidar_to_camera(points, velo2cam):
    """
    将点云从LiDAR坐标系转换到相机坐标系 (简化版，忽略r_rect)。
    Args:
        points (np.ndarray): (N, 3+) 的点云数据 (LiDAR系)。
        velo2cam (np.ndarray): 从LiDAR到相机的变换矩阵 (3x4)。

    Returns:
        np.ndarray: (N, 3) 的相机坐标系下的点云。
    """
    points_3d = points[:, :3]
    n_points = points_3d.shape[0]
    
    # 转换为齐次坐标 (N, 4)
    points_hom = np.hstack([points_3d, np.ones((n_points, 1))]).T
    
    # 直接应用 velo2cam 变换
    points_cam_hom = velo2cam @ points_hom
    
    return points_cam_hom.T

def points_in_rbox(points, rbox):
    # ... 这个函数保持不变 ...
    center = rbox[:3]
    dimensions = rbox[3:6]
    rotation_y = rbox[6]
    
    rot_mat = roty_to_mat(rotation_y)
    inv_rot_mat = rot_mat.T

    translated_points = points[:, :3] - center
    rotated_points = translated_points @ inv_rot_mat
    
    half_dims = dimensions / 2.0
    is_inside = np.all(np.abs(rotated_points) <= half_dims, axis=1)
    
    return is_inside

def roty_to_mat(rotation_y):
    # ... 这个函数保持不变 ...
    cos_ry = np.cos(rotation_y)
    sin_ry = np.sin(rotation_y)
    rot_mat = np.array([
        [cos_ry, 0, sin_ry],
        [0, 1, 0],
        [-sin_ry, 0, cos_ry]
    ])
    return rot_mat

# ==============================================================================
# --- FUNCTION TO CREATE THE DB_SAMPLER DATABASE ---
# ==============================================================================

def create_groundtruth_database(data_path, info_path, save_path, class_names):
    """
    最终修正版：修正 lidar_to_camera 的调用参数错误。
    """
    print("\n--- Creating ground truth database (Hybrid Camera-to-LiDAR Version) ---")

    database_save_path = join(save_path, 'gt_database')
    os.makedirs(database_save_path, exist_ok=True)
    
    dbinfos_save_path = join(save_path, 'drad_dbinfos_train.pkl')
    
    with open(info_path, 'rb') as f:
        train_data = pickle.load(f)

    all_db_infos = {class_name: [] for class_name in class_names}
    
    for info in tqdm(train_data['data_list'], desc="Processing frames for GT database"):
        # 1. 准备变换矩阵
        Tr_velo_to_cam_4x4 = np.array([
        [0, -1, 0, -0.4], 
        [0, 0, -1, 0.55], 
        [1, 0, 0, 0.2], 
        [0, 0, 0, 1.0]], dtype=np.float32)
        Tr_cam_to_velo_4x4 = np.linalg.inv(Tr_velo_to_cam_4x4)

        # 2. 加载点云并转换到相机系
        points_lidar = np.fromfile(join(data_path, info['lidar_points']['lidar_path']), dtype=np.float32).reshape(-1, 8)
        
        # === 核心修正：移除多余的 np.eye(4) 参数 ===
        points_cam = lidar_to_camera(points_lidar, Tr_velo_to_cam_4x4)

        # 从 .pkl 文件中读取已经处理好的（原汁原味的）相机系标注
        for i, instance in enumerate(info['instances']):
            obj_type = instance['obj_type']
            if obj_type not in class_names: continue

            # 3. 在相机系下进行匹配
            bbox_3d_cam = instance['bbox_3d']
            mask = points_in_rbox(points_cam, bbox_3d_cam)
            points_in_box_lidar = points_lidar[mask]

            # 4. 过滤掉点太少的 box
            min_points_map = {'Car': 5, 'Cyclist': 1, 'Truck': 5}
            if points_in_box_lidar.shape[0] < min_points_map.get(obj_type, 0): continue
            
            # 5. 转换标注框到雷达系，用于存入数据库
            loc_cam = bbox_3d_cam[:3]
            dims_cam = bbox_3d_cam[3:6]
            ry_cam = bbox_3d_cam[6]

            loc_cam_hom = np.append(loc_cam, 1)
            loc_lidar_hom = Tr_cam_to_velo_4x4 @ loc_cam_hom
            loc_lidar = loc_lidar_hom[:3]
            
            ry_lidar = ry_cam + (np.pi / 2)
            while ry_lidar < -np.pi: ry_lidar += 2 * np.pi
            while ry_lidar > np.pi: ry_lidar -= 2 * np.pi
            
            dims_lidar = [dims_cam[2], dims_cam[1], dims_cam[0]]
            
            bbox_3d_lidar = np.concatenate([loc_lidar, dims_lidar, [ry_lidar]]).astype(np.float32)

            # 6. 保存点云和转换后的标注
            db_filename = f"{info['sample_idx']}_{obj_type}_{i}.bin"
            points_in_box_lidar.tofile(join(database_save_path, db_filename))

            db_info = {
                'name': obj_type, 'path': join('gt_database', db_filename),
                'box3d_lidar': bbox_3d_lidar,
                'num_points_in_gt': points_in_box_lidar.shape[0],
                'difficulty': 0, 'num_point_features': 8
            }
            all_db_infos[obj_type].append(db_info)

    print("\n--- Ground truth database summary ---")
    for class_name, infos in all_db_infos.items():
        print(f"Found {len(infos)} instances of class {class_name}")
    with open(dbinfos_save_path, 'wb') as f: pickle.dump(all_db_infos, f)
    print(f"\nSuccessfully saved database info to: {dbinfos_save_path}")

# ==============================================================================
# --- MAIN SCRIPT LOGIC ---
# ==============================================================================

def create_drad_info_file(data_path, save_path, split):
    """
    为DRAD数据集创建MMDetection3D所需的信息文件 (.pkl)。
    """
    target_classes = ('Car', 'Cyclist', 'Truck')
    
    print(f"--- Processing {split} split to generate info files ---")

    image_dir = join(data_path, split, 'image_2')
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    worker_func = partial(process_single_frame, data_path=data_path, split=split, target_classes=target_classes)

    with Pool(processes=os.cpu_count()) as p:
        all_infos = list(tqdm(p.imap_unordered(worker_func, image_files), total=len(image_files), desc=f"Processing {split} files"))

    all_infos.sort(key=lambda x: x['sample_idx'])

    if split == 'training':
        train_split_idx = int(0.8 * len(all_infos))
        train_infos = all_infos[:train_split_idx]
        val_infos = all_infos[train_split_idx:]
        
        metainfo = {'categories': {name: i for i, name in enumerate(target_classes)}}
        train_data = {'metainfo': metainfo, 'data_list': train_infos}
        train_info_path = join(save_path, 'drad_infos_train.pkl')
        with open(train_info_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"Saved train infos to: {train_info_path}")
        
        val_data = {'metainfo': metainfo, 'data_list': val_infos}
        with open(join(save_path, 'drad_infos_val.pkl'), 'wb') as f:
            pickle.dump(val_data, f)
        
        print(f"Training samples: {len(train_infos)}")
        print(f"Validation samples: {len(val_infos)}")
        
        return train_info_path, target_classes

    elif split == 'testing':
        metainfo = {'categories': {name: i for i, name in enumerate(target_classes)}}
        test_data = {'metainfo': metainfo, 'data_list': all_infos}
        test_info_path = join(save_path, 'drad_infos_test.pkl')
        with open(join(save_path, 'drad_infos_test.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

        print(f"Testing samples: {len(all_infos)}")
        return None, None

# 在 tools/dataset_converters/drad_converter_new.py 文件中，替换这个函数


# 在 tools/dataset_converters/drad_converter_new.py 文件中，替换这个函数

def process_single_frame(file, data_path, split, target_classes):
    """
    处理单个数据帧并返回info字典
    (最终版：假设标签文件中的 location 已经是雷达坐标)
    """
    frame_id = file.split('.')[0]

    # 1. 标定矩阵（依然需要为 MMDetection3D 提供）
    P2_4x4 = np.array([[605.6403, 0.0, 319.2964, 0.0], [0.0, 605.6746, 235.4414, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    tr_velo_to_cam_4x4 = np.array([[0, -1, 0, -0.4], [0, 0, -1, 0.55], [1, 0, 0, 0.2], [0, 0, 0, 1.0]], dtype=np.float32)
    R0_rect_4x4 = np.eye(4, dtype=np.float32)
    calib_processed = {'P2': P2_4x4, 'Tr_velo_to_cam': tr_velo_to_cam_4x4, 'R0_rect': R0_rect_4x4}
    
    # 2. 读取标签文件
    label_path = join(data_path, split, 'label_2', frame_id + '.txt')
    annotations = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [], 'location': [], 'rotation_y': []}
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if not data or data[0] not in target_classes or len(data) < 15: continue
                annotations['name'].append(data[0])
                annotations['truncated'].append(float(data[1])); annotations['occluded'].append(int(data[2])); annotations['alpha'].append(float(data[3]))
                annotations['bbox'].append([float(x) for x in data[4:8]]); annotations['dimensions'].append([float(x) for x in data[8:11]]) # h, w, l
                annotations['location'].append([float(x) for x in data[11:14]]); annotations['rotation_y'].append(float(data[14]))

    # 3. 构建样本信息字典
    info = {'sample_idx': int(frame_id), 'images': {'CAM2': {'img_path': join(split, 'image_2', file), 'height': 480, 'width': 640, 'cam2img': P2_4x4, 'lidar2cam': tr_velo_to_cam_4x4}},
            'lidar_points': {'lidar_path': join(split, 'velodyne', frame_id + '.bin'), 'num_pts_feats': 8}, 'calib': calib_processed, 'instances': []}

    # 4. 填充标注实例
    for i in range(len(annotations['name'])):
        
        # ========================== 核心修正代码 ==========================
        # 步骤 A: 直接使用文件中的 location，并假定它就是雷达坐标
        loc_lidar = np.array(annotations['location'][i], dtype=np.float32)
        
        # 步骤 B: 将文件中的 camera rotation_y 转换为 LiDAR yaw
        # 这个转换是必要的，因为旋转的定义是相对于坐标系的轴的
        rotation_y_camera = annotations['rotation_y'][i]
        # 标准KITTI转换：雷达yaw = -相机ry - PI/2
        rotation_y_lidar = rotation_y_camera
        
        # 步骤 C: 调整尺寸顺序 (h, w, l -> l, w, h)
        dims = annotations['dimensions'][i]
        dims_lidar_order = [dims[2], dims[1], dims[0]] # l, w, h
        # ================================================================

        instance = {'bbox_label': target_classes.index(annotations['name'][i]), 'bbox_label_3d': target_classes.index(annotations['name'][i]),
                    'bbox': annotations['bbox'][i], 
                    'bbox_3d': np.concatenate([loc_lidar, dims_lidar_order, [rotation_y_lidar]]).astype(np.float32),
                    'truncated': annotations['truncated'][i], 'occluded': annotations['occluded'][i], 'alpha': annotations['alpha'][i],
                    'obj_type': annotations['name'][i], 'score': 1.0}
        info['instances'].append(instance)
    return info

if __name__ == '__main__':
    data_path = 'data/comp_data' 
    save_path = 'data/comp_data'

    print("Starting data conversion...")
    
    train_info_path, class_names = create_drad_info_file(data_path, save_path, split='training')
    
    if train_info_path:
        create_groundtruth_database(
            data_path=data_path, info_path=train_info_path,
            save_path=save_path, class_names=class_names)
    
    create_drad_info_file(data_path, save_path, split='testing')
    print("\nData conversion finished for all splits.")