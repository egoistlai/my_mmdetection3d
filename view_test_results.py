import os
import cv2
import mmcv
import pickle
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS
import torch


def draw_projected_box3d(image, corners_2d, color=(0, 255, 0), thickness=2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
    ]
    for s, e in edges:
        p1 = (int(corners_2d[s, 0]), int(corners_2d[s, 1]))
        p2 = (int(corners_2d[e, 0]), int(corners_2d[e, 1]))
        cv2.line(image, p1, p2, color, thickness)
    return image


def main():
    config_file = 'configs/drad_mvxnet/drad_mvxnet_fpn_custom.py'
    results_file = 'work_dirs/my_test_results/pred_instances.pkl'
    max_samples = 20  # 可视化数量

    print("="*50)
    print("--- Visualization of Predicted Boxes from pred_instances.pkl ---")
    print("="*50)

    # load config
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    # dataset (只需要测试集部分)
    dataset_cfg = cfg.test_dataloader.dataset
    vis_pipeline = [
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0, 1, 2, 3, 4]),
        dict(type='LoadImageFromFile'),
        dict(type='Pack3DDetInputs', keys=['points', 'img'], meta_keys=['calib', 'img_path', 'sample_idx'])
    ]
    dataset_cfg.pipeline = vis_pipeline
    dataset = DATASETS.build(dataset_cfg)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # load pred_instances.pkl
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded predictions: {len(results)} samples.")

    save_dir = "vis_preds"
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for i in range(len(dataset)):
        data_sample = dataset[i]['data_samples']
        sample_pred = results[i]

        if not hasattr(sample_pred, 'pred_instances_3d'):
            continue
        pred_sample = sample_pred.pred_instances_3d

        if pred_sample is None or len(pred_sample.bboxes_3d.tensor) == 0:
            continue

        calib = data_sample.metainfo['calib']
        P2 = np.array(calib['P2'])
        Tr_velo_to_cam = np.array(calib['Tr_velo_to_cam'])
        img_path = data_sample.metainfo['img_path']
        image = mmcv.imread(img_path)

        # corners3d_lidar: [num_boxes, 8, 3]
        corners3d_lidar = pred_sample.bboxes_3d.corners  # Tensor on GPU
        scores = pred_sample.scores_3d
        labels = pred_sample.labels_3d
        num_boxes = corners3d_lidar.shape[0]

        for j in range(num_boxes):
            box_corners = corners3d_lidar[j]  # [8,3]
            score = scores[j]
            label = labels[j]

            if score < 0.3:
                continue

            # 转齐次坐标
            corners3d_lidar_hom = torch.cat([box_corners, torch.ones((8, 1), device=box_corners.device)], dim=1)
            corners3d_lidar_hom = corners3d_lidar_hom.cpu().numpy()

            # 激光雷达到相机坐标
            corners3d_cam_hom = corners3d_lidar_hom @ Tr_velo_to_cam.T
            corners3d_cam = corners3d_cam_hom[:, :3]

            if not np.all(corners3d_cam[:, 2] > 0.1):
                continue

            # 投影到图像平面
            corners3d_cam_proj_hom = np.hstack((corners3d_cam, np.ones((8, 1))))
            corners2d_hom = corners3d_cam_proj_hom @ P2.T
            corners_2d = corners2d_hom[:, :2] / corners2d_hom[:, 2, None]

            image = draw_projected_box3d(image, corners_2d, color=(0, 0, 255), thickness=2)

        # 保存可视化结果
        output_filename = os.path.join(save_dir, f"pred_{i:06d}.png")
        cv2.imwrite(output_filename, image)
        print(f"[{count+1}] Saved -> {output_filename}")

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    print(f"\nDone! Saved {count} prediction visualizations into '{save_dir}'.")


if __name__ == '__main__':
    main()
