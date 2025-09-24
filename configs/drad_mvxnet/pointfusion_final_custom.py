# configs/drad_mvxnet/pointfusion_final_custom.py

# 1. 基础配置 (继承自您之前的设置)
_base_ = [
    '../_base_/datasets/drad_dataset.py',
    '../_base_/schedules/drad_schedule_80e.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['my_transforms.statistical_outlier_removal'],
    allow_failed_imports=False)

# 2. 点云和 Voxel 的基本定义
point_cloud_range = [-1.6, -16.0, -5.0, 32.0, 16.4, 1.0]
voxel_size = [0.15, 0.15, 0.25]

# 3. 模型定义 (最终 PointFusion 版)
model = dict(
    type='MVXFasterRCNN', # 或者 'DynamicMVXFasterRCNN'，取决于您的版本，MVXFasterRCNN 更通用

    # === 数据预处理器 (保持不变) ===
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True, voxel_type='hard', # 使用 PointFusion 时，通常用 'hard' 或 'dynamic' voxelization
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000)),
        mean=[125.4313,127.9815,127.7014], std=[57.3758, 57.1203, 58.4867],
        bgr_to_rgb=False, pad_size_divisor=32),

    # === 图像分支 (保持不变) ===
    img_backbone=dict(
        type='mmdet.ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True, style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),

    # === 点云分支：在这里实现 PointFusion ===
    pts_voxel_encoder=dict(
        type='HardVFE',  # HardVFE 是 PointFusion 的一个常见搭配
        in_channels=5,   # 您的点云维度：x,y,z,intensity,velocity
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        # === 核心：在这里嵌入您框架支持的 PointFusion ===
        fusion_layer=dict(
            type='PointFusion',
            img_channels=256,      # FPN 输出的图像特征通道数
            pts_channels=64,        # 原始点云特征通道数
            mid_channels=128,      # 中间层通道数
            out_channels=128,       # 融合后输出给 VFE 后续部分的特征通道数
            img_levels=[0, 1, 2, 3, 4], # 使用 FPN 输出的所有层级特征
            align_corners=False,
            activate_out=True,
            fuse_out=True          # 确保融合后的特征作为 VFE 的输出
        )
    ),
    
    # === 后续模块：输入通道数需要与 VFE 输出匹配 ===
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=128, # ✅ 必须与 pts_voxel_encoder.fusion_layer.out_channels 匹配
        output_shape=[216, 224]),
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=128, # ✅ 必须与 pts_middle_encoder.in_channels 匹配
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),

    # === 检测头 (保持不变) ===
    pts_bbox_head=dict(
        type='Anchor3DHead',
        in_channels=384, # 128 + 128 + 128
        # ... (您其余的 head 配置保持不变)
        num_classes=3,
        feat_channels=384,
        # use_direction_classifier=False,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [-1.0, -14.0, -1.78, 30.0, 15.0, -1.78],
                [-1.0, -14.0, -0.60, 30.0, 15.0, -0.60],
                [-1.0, -14.0, -1.78, 30.0, 15.0, -1.78],
            ],
            sizes=[ [4.74, 2.12, 1.93], [1.73, 0.73, 1.54], [9.76, 3.20, 3.18] ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.5),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)
    ),
    
    # train/test_cfg 保持不变
    train_cfg=dict(pts=dict(assigner=[
        dict(type='Max3DIoUAssigner', iou_calculator=dict(type='BboxOverlapsNearest3D'), pos_iou_thr=0.55, neg_iou_thr=0.45, min_pos_iou=0.45, ignore_iof_thr=-1),
        dict(type='Max3DIoUAssigner', iou_calculator=dict(type='BboxOverlapsNearest3D'), pos_iou_thr=0.5, neg_iou_thr=0.3, min_pos_iou=0.3, ignore_iof_thr=-1),
        dict(type='Max3DIoUAssigner', iou_calculator=dict(type='BboxOverlapsNearest3D'), pos_iou_thr=0.5, neg_iou_thr=0.35, min_pos_iou=0.35, ignore_iof_thr=-1),
    ], allowed_border=0, pos_weight=-1, debug=False)),
    test_cfg=dict(pts=dict(
        use_rotate_nms=True, nms_across_levels=False, nms_thr=0.01,
        score_thr=0.1, min_bbox_size=0, nms_pre=100, max_num=50))
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')