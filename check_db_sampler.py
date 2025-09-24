# 文件名: check_db_sampler.py (最终稳定版 - Matplotlib BEV)
import os
import pickle
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # 强制使用非GUI后端，防止任何窗口错误
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Visualize a sample from the GT Database as a BEV image.')
    parser.add_argument('--root_path', type=str, default='data/comp_data/')
    parser.add_argument('--class_name', type=str, default='Car')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='vis_db_sampler')
    args = parser.parse_args()

    # --- 1. 加载 dbinfos 文件 ---
    dbinfos_path = os.path.join(args.root_path, 'drad_dbinfos_train.pkl')
    if not os.path.exists(dbinfos_path):
        print(f"错误: 找不到 dbinfos 文件于 {dbinfos_path}"); return
    with open(dbinfos_path, 'rb') as f: dbinfos = pickle.load(f)

    # --- 2. 检查用户输入 ---
    if args.class_name not in dbinfos or not dbinfos[args.class_name]:
        print(f"错误: 在数据库中找不到或没有类别 '{args.class_name}' 的样本。"); return
    num_samples = len(dbinfos[args.class_name])
    if args.index >= num_samples:
        print(f"错误: 索引 {args.index} 超出范围。类别 '{args.class_name}' 只有 {num_samples} 个样本。"); return

    # --- 3. 获取样本信息并打印 ---
    db_info = dbinfos[args.class_name][args.index]
    print("="*50)
    print(f"正在处理样本: Class='{args.class_name}', Index={args.index}")
    print(f"点云路径: {db_info['path']}")
    print(f"点云数量: {db_info['num_points_in_gt']}")
    print(f"3D框 (雷达系): {db_info['box3d_lidar']}")
    print("="*50)

    # --- 4. 加载点云 ---
    points_path = os.path.join(args.root_path, db_info['path'])
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 8)

    # --- 5. 使用 Matplotlib 创建 BEV 可视化 ---
    box = db_info['box3d_lidar']
    center_x, center_y, _, l, w, _, yaw = box
    
    # 计算旋转后的矩形框的四个角点
    corners = np.array([
        [l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]
    ])
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    rotated_corners = corners @ rot_mat.T
    box_corners_bev = rotated_corners + np.array([center_x, center_y])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], s=5, c='blue', label='Points')
    # 绘制3D框的BEV矩形
    rect = plt.Polygon(box_corners_bev, closed=True, color='red', fill=None, linewidth=2, label='3D Box')
    ax.add_patch(rect)
    
    # 设置图像格式
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'BEV of {args.class_name} Sample {args.index}')
    ax.legend()
    ax.grid(True)

    # 保存图像
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'db_{args.class_name}_{args.index}_bev.png')
    plt.savefig(output_path)
    plt.close(fig) # 关闭图像，释放内存
    print(f"已成功保存鸟瞰图到: {output_path}")

if __name__ == '__main__':
    main()