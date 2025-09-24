import os
import cv2
import mmcv
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS


def draw_projected_box3d(image, corners_2d, color=(0, 255, 0), thickness=2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    image = image.copy()
    for start_idx, end_idx in edges:
        p1 = (int(corners_2d[start_idx, 0]), int(corners_2d[start_idx, 1]))
        p2 = (int(corners_2d[end_idx, 0]), int(corners_2d[end_idx, 1]))
        cv2.line(image, p1, p2, color, thickness)
    return image


def main():
    config_file = 'configs/drad_mvxnet/drad_mvxnet_fpn_custom.py'
    max_samples = None   # ⚠️ 控制可视化数量，比如前 10 个；设为 None 表示整个数据集
    
    print("="*50)
    print("--- .pkl File Visualization Script (Multiple Samples) ---")
    print("="*50)

    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    vis_pipeline = [
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0, 1, 2, 3, 4]),
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='Pack3DDetInputs', 
             keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
             meta_keys=['calib', 'img_path', 'sample_idx'])
    ]
    
    if cfg.train_dataloader.dataset.type == 'RepeatDataset':
        dataset_cfg = cfg.train_dataloader.dataset.dataset
    else:
        dataset_cfg = cfg.train_dataloader.dataset
    dataset_cfg.pipeline = vis_pipeline
        
    dataset = DATASETS.build(dataset_cfg)
    print(f"Dataset '{type(dataset).__name__}' loaded successfully with {len(dataset)} samples.")

    save_dir = "vis_outputs"
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for i in range(len(dataset)):
        data_info = dataset.get_data_info(i)
        if 'instances' not in data_info or len(data_info['instances']) == 0:
            continue
        
        data_package = dataset[i]
        data_sample = data_package['data_samples']
        gt_bboxes_3d_lidar = data_sample.gt_instances_3d.bboxes_3d

        if len(gt_bboxes_3d_lidar.tensor) == 0:
            continue

        calib = data_sample.metainfo['calib']
        P2 = np.array(calib['P2'])
        Tr_velo_to_cam = np.array(calib['Tr_velo_to_cam'])
        img_path = data_sample.metainfo['img_path']
        image = mmcv.imread(img_path)

        box_corners_lidar = gt_bboxes_3d_lidar.corners.numpy()
        for corners3d_lidar in box_corners_lidar:
            corners3d_lidar_hom = np.hstack((corners3d_lidar, np.ones((corners3d_lidar.shape[0], 1))))
            corners3d_cam_hom = corners3d_lidar_hom @ Tr_velo_to_cam.T
            corners3d_cam = corners3d_cam_hom[:, :3]

            if not np.all(corners3d_cam[:, 2] > 0.1):
                continue  # skip if behind camera

            corners3d_cam_proj_hom = np.hstack((corners3d_cam, np.ones((corners3d_cam.shape[0], 1))))
            corners2d_hom = corners3d_cam_proj_hom @ P2.T
            corners_2d = corners2d_hom[:, :2] / corners2d_hom[:, 2, None]

            image = draw_projected_box3d(image, corners_2d)

        # 保存图片
        output_filename = os.path.join(save_dir, f"sample_{i:06d}.png")
        cv2.imwrite(output_filename, image)
        print(f"[{count+1}] Saved visualization -> {output_filename}")

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    print(f"\nDone! Saved {count} visualizations into '{save_dir}'.")


if __name__ == '__main__':
    main()
