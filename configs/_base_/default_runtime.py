# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None   # [origin]: load_from = None
# load_from = "/data1/users/hanshuaihao01/mmrotate/work_dirs/oriented_reppoint/1018_exp1_lr_0.004_fem5_con4/epoch_14.pth"   # [origin]: load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
