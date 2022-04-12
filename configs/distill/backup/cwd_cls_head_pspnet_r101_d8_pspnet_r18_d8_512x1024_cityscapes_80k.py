dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
student = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
teacher = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model=dict(
            type='mmseg.EncoderDecoder',
            backbone=dict(
                type='ResNetV1c',
                init_cfg=dict(
                    type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
                depth=18,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
            decode_head=dict(
                type='PSPHead',
                in_channels=512,
                in_index=3,
                channels=128,
                pool_scales=(1, 2, 3, 6),
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0)),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=256,
                in_index=2,
                channels=64,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.4)),
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=dict(
            type='mmseg.EncoderDecoder',
            backbone=dict(
                type='ResNetV1c',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
            decode_head=dict(
                type='PSPHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                pool_scales=(1, 2, 3, 6),
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0)),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=
                'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
            )),
        teacher_trainable=False,
        components=[
            dict(
                student_module='decode_head.conv_seg',
                teacher_module='decode_head.conv_seg',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_logits',
                        tau=1,
                        loss_weight=5)
                ])
        ]))
find_unused_parameters = True
work_dir = 'work_dir'
auto_resume = False
gpu_ids = [0]
