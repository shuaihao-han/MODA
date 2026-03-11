# Rotation robustness eval config:
# - rotates the input image by a fixed angle (deg) via env var `ROT_ANGLE`
# - ground-truth labels should be rotated correspondingly by the driver script
#
# NOTE: Child/base configs are executed in separate modules, so we redefine
# the small set of constants we need (img_norm_cfg, angle_version) locally.

_base_ = ['./start_level_0.py']

import os as _os

angle_version = 'le135'
img_norm_cfg = dict(
    mean=[81.59, 82.86, 81.93, 79.98, 82.54, 79.70, 81.44, 79.82],
    std=[39.67, 41.84, 37.99, 39.47, 43.50, 32.80, 38.43, 37.50],
    to_rgb=False)

_ROT_ANGLE = float(_os.getenv('ROT_ANGLE', '0'))

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
            dict(type='FixedRotate', angle=_ROT_ANGLE),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

# Keep the dataset paths from the base dataset config, but override the
# pipeline to include FixedRotate. `rotation_sweep_eval.py` will override
# `data.test.ann_file` per angle.
data = dict(
    val=dict(pipeline=test_pipeline, version=angle_version),
    test=dict(pipeline=test_pipeline, version=angle_version),
)
