# configs/_base_/datasets/drad_dataset.py

# 1. 数据集相关的基础变量
dataset_type = 'DRADDataset'
data_root = 'data/comp_data/'
class_names = ['Car', 'Cyclist', 'Truck']
point_cloud_range = [-1.6, -16.0, -5.0, 32.0, 16.4, 1.0] # <-- 使用新范围
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
backend_args = None

# 2. DbSampler 的定义
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'drad_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Cyclist=1, Truck=5)),
    classes=class_names,
    sample_groups=dict(Car=5, Cyclist=6, Truck=3), # <-- 建议使用温和的增强策略
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=[0, 1, 2, 3, 4]))

# 3. 训练数据处理流水线
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    # dict(
    #     type='Det3DDataPreprocessor',
    #     mean=[15.2, 0.2, -2.0, 128.0, 0.5],  # 你的点云5个维度的均值
    #     std=[16.8, 16.2, 3.0, 64.0, 0.5]     # 你的点云5个维度的标准差
    # ),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='StatisticalOutlierRemoval', nb_neighbors=20, std_ratio=2.0), # (可选)
    dict(type='ObjectSample', db_sampler=db_sampler), # <-- 重新启用，但数量减少
    dict(
        type='ObjectNoise',
        # ✅ 根据您的框架定义，使用正确的参数名 num_try
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# 4. 测试数据处理流水线
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=8,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans', rot_range=[0, 0],
                scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# 5. Dataloader 的定义 (修正了 val 和 test 的 data_prefix)
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type, data_root=data_root,
            ann_file='drad_infos_train.pkl',
            pipeline=train_pipeline, modality=input_modality,
            metainfo=metainfo, filter_empty_gt=False,
            box_type_3d='LiDAR', backend_args=backend_args,
            data_prefix=dict(pts='', img='')))) # <-- 确保为空

val_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file='drad_infos_val.pkl',
        pipeline=test_pipeline, modality=input_modality,
        metainfo=metainfo, test_mode=True, box_type_3d='LiDAR',
        backend_args=backend_args,
        data_prefix=dict(pts='', img=''))) # <-- 修正为空

test_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file='drad_infos_test.pkl',
        pipeline=test_pipeline, modality=input_modality,
        metainfo=metainfo, test_mode=True, box_type_3d='LiDAR',
        backend_args=backend_args,
        data_prefix=dict(pts='', img=''))) # <-- 修正为空

val_evaluator = dict(
    type='KittiMetric', ann_file=data_root + 'drad_infos_val.pkl',
    backend_args=backend_args)

test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'drad_infos_test.pkl',
    backend_args=backend_args)