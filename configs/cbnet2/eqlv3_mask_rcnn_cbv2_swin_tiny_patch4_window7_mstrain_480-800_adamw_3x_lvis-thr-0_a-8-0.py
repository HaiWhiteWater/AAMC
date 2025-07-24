# model settings

model = dict(
    type='MaskRCNN',
    pretrained=None,  # Set to None, need to manually specify location during training
    backbone=dict(
        type='CBSwinTransformer',
        # type='SwinTransformerOriginal',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='CBFPN',  # +
        # type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1203,  # 80
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='EQLV3Loss',
                use_sigmoid=True,
                loss_weight=1.0,
                alpha=8),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1203,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,  #
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,  # 100 in COCO, changed to 300 for LVIS
            mask_thr_binary=0.5)))

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR, except for size_divisor=32
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
            [
                dict(
                    type='Resize',
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# dataloader
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    # workers_per_gpu=4,
    # workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix=data_root,
            pipeline=train_pipeline,
            # filter_cfg=dict(filter_empty_gt=True, min_size=32)
        )    ),  # lvis_v1_instance.py in new version of mmdet
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=6, metric=['bbox', 'segm'])

# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    # lr=0.000075,  # VIDeT 0.0002 and 0.0001  4gpu:0.00005 6gpu:0.000075
    lr=0.0001,  # VIDeT 0.0002 and 0.0001  4gpu:0.00005 6gpu:0.000075
    betas=(0.9, 0.999),  # Hyperparameter for gradient computation
    weight_decay=0.05,
    # VIDeT 0.1 recommended weight_decay values are usually between 0.01 to 0.1, too high values may cause underfitting# Weight decay parameter in AdamW optimizer, controls regularization strength. Larger values increase regularization and help reduce overfitting
    paramwise_cfg=dict(  # Dictionary containing custom configuration for parameter weight decay
        custom_keys={  # Includes definitions of weight decay multipliers (decay_mult) for different model parameters, set to 0 means they won't be affected by weight decay, i.e., no regularization
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=None)  # Gradient clipping. grad_clip is usually used to control gradient magnitude to prevent gradient explosion


fp16 = dict(loss_scale='dynamic')  # New addition. CBNet official implementation doesn't use this, possibly due to low version

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,  # VITDet 250
    warmup_ratio=0.001,
    step=[27, 33]
    # step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=36)


# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='UpdateAlphaHook',
         update_interval=1,
         decay_rate=0.225)
]

# default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=12))
# default_hooks = dict(checkpoint=dict(type='CheckpointHook', save_best='auto'))
# Create weights for release
# default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='accuracy', rule='less', published_keys=['meta', 'state_dict']))
# checkpoint_config = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
# checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
# auto_scale_lr = dict(enable=False, base_batch_size=16)


