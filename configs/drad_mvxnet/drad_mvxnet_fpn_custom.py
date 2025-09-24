# configs/drad_mvxnet/drad_mvxnet_fpn_custom.py

# 继承数据集配置、训练策略和默认runtime
_base_ = [
    '../_base_/datasets/drad_dataset.py',
    '../_base_/schedules/drad_schedule_80e.py',
    '../_base_/default_runtime.py'
]

# 自定义模块导入
custom_imports = dict(
    imports=['my_transforms.statistical_outlier_removal'],
    allow_failed_imports=False
)

# 点云范围 & voxel size
point_cloud_range = [-1.6, -16.0, -5.0, 32.0, 16.4, 1.0]
voxel_size = [0.15, 0.15, 0.25]

# 模型定义
model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='hard',
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000)),
        mean=[125.4313,127.9815,127.7014],
        std=[57.3758, 57.1203, 58.4867],
        bgr_to_rgb=False,
        pad_size_divisor=32),

    # 图像 backbone + neck
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    # 点云部分
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # H=216
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])   # W=224
    ]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),   # 输出通道 = 128+128+128=384

    # BBox Head
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,          # ✅ 保证和 neck 输出一致
        feat_channels=384,        # 中间卷积层通道
        use_direction_classifier=False,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            # Anchor 区域与点云范围匹配
            ranges=[
                [-1.0, -14.0, -1.78, 30.0, 15.0, -1.78],  # Car
                [-1.0, -14.0, -0.60, 30.0, 15.0, -0.60],  # Cyclist
                [-1.0, -14.0, -1.78, 30.0, 15.0, -1.78],  # Truck
            ],
            sizes=[
                [4.74, 2.12, 1.93],   # Car
                [1.73, 0.73, 1.54],   # Cyclist
                [9.76, 3.20, 3.18]    # Truck
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(
            type='DeltaXYZWLHRBBoxCoder',
            target_norm_cfg=dict(
                mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # 建议添加归一化
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.0)),

    # 训练配置
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50))
)


# checkpoint 配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,      # 每 5 个 epoch 保存一次
        save_best='Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_moderate',
        rule='greater',
        max_keep_ckpts=3
    )
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
# 不加载预训练，从头开始
load_from = None
