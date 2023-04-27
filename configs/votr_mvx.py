_base_ = ['../configs/_base_/default_runtime.py']

# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 100, 40, 1]

model = dict(
    type='DynamicMVXFasterRCNN',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type="Pretrained", checkpoint="/media/ntu/volume1/home/s122md304_13/quan_fyp/pretrained_img_backbone/resnet50_caffe-788b5fa3.pth"
        ),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type='PointFusion',
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False)),
    pts_middle_encoder=dict(
        type='VoxelTransformer',
        in_channels=128,
        sparse_shape=[2000, 1600, 41],
        voxel_size=[0.05, 0.05, 0.1],
        use_relative_coords=True,
        use_pooled_feature=True,
        use_no_query_coords=True,
        num_point_features=64,
        hash_size=400000,
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        layers_list=[
            dict(
                SP_CFGS=dict(
                    CHANNELS=[16, 32, 32],
                    DROPOUT=0,
                    NUM_HEADS=4,
                    ATTENTION=[
                        dict(
                            NAME='StridedAttention',
                            SIZE=48,
                            RANGE_SPEC=[[0, 2, 1, 0, 2, 1, 0, 2, 1],
                                        [2, 5, 1, 2, 5, 1, 0, 3, 1],
                                        [5, 25, 5, 5, 25, 5, 0, 15, 2],
                                        [25, 125, 25, 25, 125, 25, 0, 15, 3]])
                    ],
                    STRIDE=[2, 2, 2],
                    NUM_DS_VOXELS=90000),
                SUBM_CFGS=dict(
                    NUM_BLOCKS=2,
                    CHANNELS=[32, 32, 32],
                    DROPOUT=0,
                    NUM_HEADS=4,
                    ATTENTION=[
                        dict(
                            NAME='StridedAttention',
                            SIZE=48,
                            RANGE_SPEC=[[0, 2, 1, 0, 2, 1, 0, 2, 1],
                                        [2, 4, 1, 2, 4, 1, 0, 3, 1],
                                        [4, 12, 3, 4, 12, 3, 0, 8, 2],
                                        [12, 60, 12, 12, 60, 12, 0, 8, 2]])
                    ],
                    USE_POS_EMB=True)),
            dict(
                SP_CFGS=dict(
                    CHANNELS=[32, 64, 64],
                    DROPOUT=0,
                    NUM_HEADS=4,
                    ATTENTION=[
                        dict(
                            NAME='StridedAttention',
                            SIZE=48,
                            RANGE_SPEC=[[0, 2, 1, 0, 2, 1, 0, 2, 1],
                                        [2, 4, 1, 2, 4, 1, 0, 3, 1],
                                        [4, 12, 3, 4, 12, 3, 0, 8, 2],
                                        [12, 60, 12, 12, 60, 12, 0, 8, 2]])
                    ],
                    STRIDE=[2, 2, 2],
                    NUM_DS_VOXELS=90000),
                SUBM_CFGS=dict(
                    NUM_BLOCKS=2,
                    CHANNELS=[64, 64, 64],
                    DROPOUT=0,
                    NUM_HEADS=4,
                    ATTENTION=[
                        dict(
                            NAME='StridedAttention',
                            SIZE=48,
                            RANGE_SPEC=[[0, 2, 1, 0, 2, 1, 0, 2, 1],
                                        [2, 3, 1, 2, 3, 1, 0, 2, 1],
                                        [3, 8, 2, 3, 8, 2, 0, 4, 1],
                                        [8, 32, 8, 8, 32, 8, 0, 4, 1]])
                    ],
                    USE_POS_EMB=True)),
            dict(
                SP_CFGS=dict(
                    CHANNELS=[64, 64, 64],
                    DROPOUT=0,
                    NUM_HEADS=4,
                    ATTENTION=[
                        dict(
                            NAME='StridedAttention',
                            SIZE=48,
                            RANGE_SPEC=[[0, 2, 1, 0, 2, 1, 0, 2, 1],
                                        [2, 3, 1, 2, 3, 1, 0, 2, 1],
                                        [3, 8, 2, 3, 8, 2, 0, 4, 1],
                                        [8, 32, 8, 8, 32, 8, 0, 4, 1]])
                    ],
                    STRIDE=[2, 2, 2],
                    NUM_DS_VOXELS=90000),
                SUBM_CFGS=dict(
                    NUM_BLOCKS=2,
                    CHANNELS=[64, 64, 64],
                    DROPOUT=0,
                    NUM_HEADS=4,
                    ATTENTION=[
                        dict(
                            NAME='StridedAttention',
                            SIZE=48,
                            RANGE_SPEC=[[0, 2, 1, 0, 2, 1, 0, 2, 1],
                                        [2, 4, 1, 2, 4, 1, 0, 3, 1],
                                        [4, 16, 2, 4, 16, 2, 0, 5, 1]])
                    ],
                    USE_POS_EMB=True))
        ]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=320,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6, 100, 40.0, -0.6],
                [0, -40.0, -0.6, 100, 40.0, -0.6],
                [0, -40.0, -1.78, 100, 40.0, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
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
        pts=dict(
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
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

# dataset settings
dataset_type = 'KittiDataset'
data_root = '/media/ntu/volume1/home/s122md304_13/quan_fyp/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality = dict(use_lidar=True, use_camera=True)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='Resize',
        img_scale=[(640, 192), (2560, 768)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

# Training settings
optimizer = dict(weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

evaluation = dict(interval=1, pipeline=eval_pipeline)

lr = 0.0001  # max learning rate
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

momentum_config = None

runner = dict(type='EpochBasedRunner', max_epochs=100)
work_dir = "/media/ntu/volume1/home/s122md304_13/quan_fyp/train_kitti/votr_mvx_cont"

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

resume_from = '/media/ntu/volume1/home/s122md304_13/quan_fyp/train_kitti/votr_mvx/epoch_45.pth'