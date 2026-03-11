_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RotatedRepPointsMSI',
    backbone=dict(
        type='ResNet',
        in_channels=8,    # [added]
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        with_cp=True,   # [added] 默认为False用来控制是否使用checkpointing来节省显存,但是会增加计算开销,增加训练时间
        # frozen_stages=1,    # [origin]: frozen_stages=1(冻结第一个stage的参数)
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='work_dirs/pretrained_weight/0706_resnet50_fine_tune_train_exp1_step2_2.pth')),   # [origin]: checkpoint='torchvision://resnet50'
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,   # 从第几个输入特征图开始构建 FPN 金字塔, start_level=1 意味着从输入的第二个特征图（即 C3, 通道数为 512）开始构建金字塔
        add_extra_convs='on_input',   # 在最后一个特征图（C5）之后添加额外的卷积层, 以生成更多层次的特征图
        num_outs=5,      # 经过FPN后的输出是5个层次的特征图
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=8,  # [origin]: 8
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        init_qua_weight=0.2,
        top_ratio=0.4),
    # training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxConvexIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.4),
        max_per_img=2000))

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)   # NOTE：对于rgb图像 to_rgb 参数要设置为True, 对于MSI图像设置为False
img_norm_cfg = dict(
    mean=[81.59, 82.86, 81.93, 79.98, 82.54, 79.70, 81.44, 79.82], std=[39.67, 41.84, 37.99, 39.47, 43.50, 32.80, 38.43, 37.50], to_rgb=False)
# 谱导数
# img_norm_cfg = dict(
#     mean=[81.59, 82.86, 81.93, 79.98, 82.54, 79.70, 81.44, 79.82, 1.28, -0.93, -1.95, 2.56, -2.85, 1.74, -1.62, 24.11], 
#     std=[39.67, 41.84, 37.99, 39.47, 43.50, 32.80, 38.43, 37.50, 14.54, 13.77, 11.21, 9.29, 34.61, 16.11, 7.3, 22.73], to_rgb=False)
# img_norm_cfg = dict(
#     mean=[81.59, 82.86, 81.93, 79.98, 82.54, 79.70, 81.44, 79.82, 184.05, 184.05, 184.05, 184.05, 184.05, 184.05, 184.05, 184.05], 
#     std=[39.67, 41.84, 37.99, 39.47, 43.50, 32.80, 38.43, 37.50, 54.89, 54.89, 54.89, 54.89, 54.89, 54.89, 54.89, 54.89], to_rgb=False)
# img_norm_cfg = dict(
#     mean=[81.59, 82.86, 81.93, 79.98, 82.54, 79.70, 81.44, 79.82, 108.20, 108.20, 108.20, 108.20, 108.20, 108.20, 108.20, 108.20], 
#     std=[39.67, 41.84, 37.99, 39.47, 43.50, 32.80, 38.43, 37.50, 59.93, 59.93, 59.93, 59.93, 59.93, 59.93, 59.93, 59.93], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1200, 1200)),   # [origin]: img_scale=(1024, 1024)
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),   # 对图像添加padding, 使得图像尺寸H, W能够被32整除
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 1200),   # [origin]: img_scale=(1024, 1024)
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(pipeline=test_pipeline, version=angle_version),
    test=dict(pipeline=test_pipeline, version=angle_version))
optimizer = dict(lr=0.004)   # [origin]: lr=0.008 
# optimizer = dict(lr=0.000)  # 暂时不使用优化器
checkpoint_config = dict(interval=1, max_keep_ckpts=7)   # [origin]: checkpoint_config = dict(interval=1)  设置最多保存4个checkpoints
evaluation = dict(interval=2, metric='mAP')  # evaluation = dict(interval=1, metric='mAP')
# fp16 settings
# fp16 = dict(loss_scale='dynamic')   # 设置混合精度训练，loss_scale='dynamic'表示动态调整loss scale
# 防止未产生损失的参数引发问题
# find_unused_parameters = True
