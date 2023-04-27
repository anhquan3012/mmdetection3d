_base_ = [
    '../configs/_base_/schedules/cyclic_40e.py', 
    '../configs/_base_/default_runtime.py'
]

voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 100, 40, 1]

model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='VoxelTransformer',
        in_channels=4,
        sparse_shape=[2000, 1600, 41],
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
                    DROPOUT = 0,
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
                    DROPOUT = 0,
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
                    DROPOUT = 0,
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
                    USE_POS_EMB = True
                )
            ),
            dict(
                SP_CFGS = dict(
                    CHANNELS = [64, 64, 64],
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
        in_channels=320,
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
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -1.78, 100, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
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


dataset_type = 'SimulatedDataset'
data_root = '/media/ntu/volume1/home/s122md304_13/quan_fyp/simulated_data/kitti_format_64/'
class_names = ['Car']
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'simulated_train.pkl',
            split='training',
            pts_prefix='velodyne',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'simulated_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=None,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'simulated_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=None,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR')
)

evaluation = dict(interval=60)

work_dir = '/media/ntu/volume1/home/s122md304_13/quan_fyp/train_simulated/votr_car_short_13Apr'

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

lr = 1e-4
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='cyclic',
    target_ratio=(3, 1e-2),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

runner = dict(type='EpochBasedRunner', max_epochs=40)
