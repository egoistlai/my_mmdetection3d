# 文件名: final_data_validator.py
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Final Validator for raw data alignment.')
    parser.add_argument('--frame_id', type=str, default='000071', help='Frame ID to validate (e.g., 000071).')
    parser.add_argument('--data_path', type=str, default='data/comp_data/training', help='Path to the training data folder.')
    args = parser.parse_args()

    frame_id = args.frame_id
    data_path = args.data_path
    
    print("="*60)
    print(f"--- 最终诊断脚本：验证帧 {frame_id} ---")
    print("="*60)

    # --- 1. 加载所有原始文件 ---
    lidar_path = os.path.join(data_path, 'velodyne', f'{frame_id}.bin')
    label_path = os.path.join(data_path, 'label_2', f'{frame_id}.txt')
    calib_path = os.path.join(data_path, 'calib', f'{frame_id}.txt')

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 8)
    
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    Tr_velo_to_cam = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape(3, 4)

    with open(label_path, 'r') as f:
        # 只读取第一个Car的标签进行验证
        first_car_line = None
        for line in f:
            if line.strip().split(' ')[0] == 'Car':
                first_car_line = line.strip().split(' ')
                break
    
    if not first_car_line:
        print(f"错误：在 {label_path} 中未找到 'Car' 标签。")
        return

    # --- 2. 解析原始数据 ---
    dims_cam = np.array([float(d) for d in first_car_line[8:11]])  # h, w, l
    loc_cam = np.array([float(d) for d in first_car_line[11:14]]) # x, y, z (相机系)
    ry_cam = float(first_car_line[14]) # rotation_y (相机系)

    print(f"从标定文件 {frame_id}.txt 读取的 Tr_velo_to_cam:\n{Tr_velo_to_cam}\n")
    print(f"从标签文件 {frame_id}.txt 读取的相机系 Location: {loc_cam}")
    print(f"从标签文件 {frame_id}.txt 读取的相机系 Rotation Y: {ry_cam}\n")

    # --- 3. 执行我们认为正确的“相机->雷达”转换 ---
    # 构造4x4变换矩阵
    Tr_velo_to_cam_4x4 = np.array([[0, -1, 0, -0.4], [0, 0, -1, 0.55], [1, 0, 0, 0.2], [0, 0, 0, 1.0]], dtype=np.float32)
    # 计算逆矩阵
    Tr_cam_to_velo_4x4 = np.linalg.inv(Tr_velo_to_cam_4x4)
    
    # 转换 location
    loc_cam_hom = np.append(loc_cam, 1)
    loc_lidar_hom = Tr_cam_to_velo_4x4 @ loc_cam_hom
    loc_lidar = loc_lidar_hom[:3]

    # 转换 rotation_y
    ry_lidar = ry_cam - (np.pi / 2)
    while ry_lidar < -np.pi: ry_lidar += 2 * np.pi
    while ry_lidar > np.pi: ry_lidar -= 2 * np.pi

    # 调整尺寸顺序
    dims_lidar = np.array([dims_cam[2], dims_cam[1], dims_cam[0]]) # l, w, h
    
    print(f"计算出的雷达系 Location: {loc_lidar}")
    print(f"计算出的雷达系 Rotation Y: {ry_lidar}\n")

    # --- 4. 绘制鸟瞰图进行可视化验证 ---
    center_x, center_y, _, l, w, _, yaw = np.concatenate([loc_lidar, dims_lidar, [ry_lidar]])
    
    corners = np.array([
        [l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]
    ])
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated_corners = corners @ rot_mat.T
    box_corners_bev = rotated_corners + np.array([center_x, center_y])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(points[:, 0], points[:, 1], s=1, c='gray', label='Full Point Cloud')
    rect = plt.Polygon(box_corners_bev, closed=True, color='red', fill=None, linewidth=2, label='Transformed GT Box')
    ax.add_patch(rect)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X - Forward (m)')
    ax.set_ylabel('Y - Left (m)')
    ax.set_title(f'Validation for Frame {frame_id} - Are they aligned?')
    ax.legend()
    ax.grid(True)

    output_path = f'validation_bev_{frame_id}.png'
    plt.savefig(output_path)
    print(f"诊断图像已保存到: {output_path}")

if __name__ == '__main__':
    main()