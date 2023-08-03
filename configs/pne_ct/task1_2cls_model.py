fp16 = dict(loss_scale=512.)
pretrained ='mask_rcnn_cdswinlepe3dpvtv2_377_rear_0001_stiny_direct_finetune_1x_coco_fp16_mAP375.pth'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinLePETransformer3D',
        embed_dims=32,
        patch_size=(4,4,4),
        in_channels=1,
        strides=((4,4,4), (2,2,2), (2,2,2), (2,2,2)),
        window_size=(4,4,4),
        depths=[1, 2, 11, 1],
        num_heads=[2, 4, 8, 16],
        parallel_group=False,
        out_indices=[3],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        mixed_arch=True,
        pvt_stage=[False, False, False, True],
        pvt_pos='rear',
        use_conv_ffn=True,
        sr_ratios=[8, 4, 2, 1],
        fixz_sr_ratios=[3, 3, 3, 1], 
        output_2d=False,
        with_cp=False,
        convert_weights=False,
        pretrained2d=False,
        init_cfg=None),
    neck=dict(type='GlobalAveragePooling', dim=3),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        multi_cls=True
    ))
# dataset settings
dataset_type = 'huaxiMultiPneuCTClsDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    # 1. random crop 2. reflection(flip by 3 axis)
    dict(type='LoadTensorFromFile', data_keys='data', transpose=True, to_float32=True),
    dict(
        type='PhotoMetricDistortionMultipleSlices',
        brightness_delta=32,
        contrast_range=(0.8, 1.2)),
    dict(type='TensorNormCropFlip', crop_size=(224, 224, 64), move=8, train=True, mean=img_norm_cfg['mean'][0], std=img_norm_cfg['std'][0]),
    dict(type='ToTensor', keys=['img'],transpose=True),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadTensorFromFile', data_keys='data', transpose=True, to_float32=True),
    dict(type='TensorNormCropFlip', crop_size=(224, 224, 64), move=8, train=False, mean=img_norm_cfg['mean'][0], std=img_norm_cfg['std'][0]),
    dict(type='ToTensor', keys=['img'],transpose=True),
    dict(type='Collect', keys=['img'])
]
root_dir = '/home/majiechao/code/CAMAS/mmi/data/'
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    num_per_ct=32,
    pos_fraction = 0.5,
    train=dict(
        type=dataset_type,
        data_prefix=root_dir + 'sample_data',
        ann_file= root_dir + 'meta/task1_pne_2cls_train.csv',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root_dir + 'sample_data',
        ann_file= root_dir + 'meta/task1_pne_2cls_val.csv',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=root_dir + 'sample_data',
        ann_file= root_dir + 'meta/task1_pne_2cls_train.csv',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='auc')


# checkpoint saving
checkpoint_config = dict(interval=6)
# yapf:disable
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=60)
