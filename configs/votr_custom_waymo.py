_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../configs/_base_/default_runtime.py'
]

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [0, -51.2, -2, 76.8, 51.2, 4]
input_modality = dict(use_lidar=True, use_camera=False)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4]))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=5)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='VoxelTransformer',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        voxel_size=voxel_size,
        use_relative_coords = True,
        use_pooled_feature = True,
        use_no_query_coords = True,
        num_point_features = 64,
        hash_size = 400000,
        point_cloud_range = point_cloud_range,
        layers_list = [
            dict(
                SP_CFGS = dict(
                    CHANNELS = [16, 32, 32],
                    DROPOUT = 0.4,
                    NUM_HEADS = 4,
                    ATTENTION = [dict(
                        NAME = 'StridedAttention',
                        SIZE = 48,
                        RANGE_SPEC = [[0, 2, 1, 0, 2, 1, 0, 2, 1], 
                                      [2, 5, 1, 2, 5, 1, 0, 3, 1], 
                                      [5, 25, 5, 5, 25, 5, 0, 15, 2], 
                                      [25, 125, 25, 25, 125, 25, 0, 15, 3]],
                    )],
                    STRIDE = [2, 2, 2],
                    NUM_DS_VOXELS = 90000
                ),
                SUBM_CFGS = dict(
                    NUM_BLOCKS = 2,
                    CHANNELS = [32, 32, 32],
                    DROPOUT = 0.4,
                    NUM_HEADS = 4,
                    ATTENTION = [dict(
                        NAME = 'StridedAttention',
                        SIZE = 48,
                        RANGE_SPEC = [[0, 2, 1, 0, 2, 1, 0, 2, 1], 
                                      [2, 4, 1, 2, 4, 1, 0, 3, 1], 
                                      [4, 12, 3, 4, 12, 3, 0, 8, 2], 
                                      [12, 60, 12, 12, 60, 12, 0, 8, 2]],
                    )],
                    USE_POS_EMB = True
                )
            ),
            dict(
                SP_CFGS = dict(
                    CHANNELS = [32, 64, 64],
                    DROPOUT = 0.2,
                    NUM_HEADS = 4,
                    ATTENTION = [dict(
                        NAME = 'StridedAttention',
                        SIZE = 48,
                        RANGE_SPEC = [[0, 2, 1, 0, 2, 1, 0, 2, 1], 
                                      [2, 4, 1, 2, 4, 1, 0, 3, 1], 
                                      [4, 12, 3, 4, 12, 3, 0, 8, 2], 
                                      [12, 60, 12, 12, 60, 12, 0, 8, 2]],
                    )],
                    STRIDE = [2, 2, 2],
                    NUM_DS_VOXELS = 90000
                ),
                SUBM_CFGS = dict(
                    NUM_BLOCKS = 2,
                    CHANNELS = [64, 64, 64],
                    DROPOUT = 0.2,
                    NUM_HEADS = 4,
                    ATTENTION = [dict(
                        NAME = 'StridedAttention',
                        SIZE = 48,
                        RANGE_SPEC = [[0, 2, 1, 0, 2, 1, 0, 2, 1], 
                                      [2, 3, 1, 2, 3, 1, 0, 2, 1], 
                                      [3, 8, 2, 3, 8, 2, 0, 4, 1], 
                                      [8, 32, 8, 8, 32, 8, 0, 4, 1]],
                    )],
                    USE_POS_EMB = True
                )
            ),
            dict(
                SP_CFGS = dict(
                    CHANNELS = [32, 64, 64],
                    DROPOUT = 0,
                    NUM_HEADS = 4,
                    ATTENTION = [dict(
                        NAME = 'StridedAttention',
                        SIZE = 48,
                        RANGE_SPEC = [[0, 2, 1, 0, 2, 1, 0, 2, 1], 
                                      [2, 3, 1, 2, 3, 1, 0, 2, 1], 
                                      [3, 8, 2, 3, 8, 2, 0, 4, 1], 
                                      [8, 32, 8, 8, 32, 8, 0, 4, 1]],
                    )],
                    STRIDE = [2, 2, 2],
                    NUM_DS_VOXELS = 90000
                ),
                SUBM_CFGS = dict(
                    NUM_BLOCKS = 2,
                    CHANNELS = [64, 64, 64],
                    DROPOUT = 0,
                    NUM_HEADS = 4,
                    ATTENTION = [dict(
                        NAME = 'StridedAttention',
                        SIZE = 48,
                        RANGE_SPEC = [[0, 2, 1, 0, 2, 1, 0, 2, 1], 
                                      [2, 4, 1, 2, 4, 1, 0, 3, 1], 
                                      [4, 16, 2, 4, 16, 2, 0, 5, 1]],
                    )],
                    USE_POS_EMB = True
                )
            )
        ]
    ),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

work_dir = '/media/ntu/volume1/home/s122md304_13/quan_fyp/train_kitti/votr'

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

runner = dict(type='EpochBasedRunner', max_epochs=80)
