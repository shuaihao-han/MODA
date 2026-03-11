# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)     # 如果使用AdamW优化器，将其注释掉，在配置文件中进行修改
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],    # [origin]: step=[8, 11]
    gamma=0.1)   # [origin]: gamma=0.1
runner = dict(type='EpochBasedRunner', max_epochs=12)  # [origin]: max_epochs=12
checkpoint_config = dict(interval=1)
