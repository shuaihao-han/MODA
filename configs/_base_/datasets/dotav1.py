# dataset settings
dataset_type = 'DOTADataset'
data_root = 'data/mod/'   # [origin]: data_root = 'data/split_1024_dota1_0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1200, 1200)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 1200),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
# 将train_set与test_set设置相同,先进行自训自检看是否能正常训练
data = dict(
    samples_per_gpu=2,   # [origin]: samples_per_gpu=2,
    workers_per_gpu=2,   # [origin: workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labels/',   # [origin]: ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'train/images/',   # [origin]: img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labels/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labels/',   # ann_file=data_root + 'test/labels_vis/',
        img_prefix=data_root + 'test/images/',  #img_prefix=data_root + 'test/images_vis/',
        pipeline=test_pipeline))
