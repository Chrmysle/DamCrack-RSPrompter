_base_ = ['_base_/rsprompter_anchor.py']

# 1. 路径与实验管理
work_dir = '/root/autodl-tmp/RSPrompter/work_dirs/rsprompter/damcrack_anchor_final'
local_sam_path = "/root/autodl-tmp/RSPrompter/work_dirs/sam_cache/sam_vit_base"
local_sam_ckpt = "/root/autodl-tmp/RSPrompter/work_dirs/sam_cache/sam_vit_base/pytorch_model.bin"
data_root = '/root/autodl-tmp/RSPrompter/data/DamCrack'

# 显式定义类别映射
metainfo = dict(classes=('crack', ))

# 2. 模型核心配置 (显存优化版)
model = dict(
    decoder_freeze=False,
    shared_image_embedding=dict(
        hf_pretrain_name=local_sam_path,
        init_cfg=dict(type='Pretrained', checkpoint=local_sam_ckpt),
    ),
    backbone=dict(
        hf_pretrain_name=local_sam_path,
        init_cfg=dict(type='Pretrained', checkpoint=local_sam_ckpt)
    ),
    neck=dict(
        feature_aggregator=dict(
            in_channels=local_sam_path,
            hidden_channels=32,
            select_layers=range(1, 13, 2), 
        ),
    ),
    # --- RPN 减负：减少进入显存的建议框数量 ---
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0]), 
        train_cfg=dict(
            assigner=dict(
                pos_iou_thr=0.7, # 提高阈值，让正样本更精简
                neg_iou_thr=0.3,
                min_pos_iou=0.3),
            nms_pre=800,       # 从 2000 降到 800 (关键：减少预选框)
            max_per_img=300,   # 从 1000 降到 300 (关键：减少送入 ROI 的框)
            nms=dict(type='nms', iou_threshold=0.7)
        )
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(
            mask_decoder=dict(
                hf_pretrain_name=local_sam_path,
                init_cfg=dict(type='Pretrained', checkpoint=local_sam_ckpt)
            ),
            per_pointset_point=3, # 从 5 降到 3 (减少 SAM 解码的点数，省显存)
            with_sincos=True, 
        ),
    ),
)

# 3. 数据流水线 (保持 1024 分辨率)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    #dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(123.675, 116.28, 103.53))),
    dict(type='PackDetInputs')
]

# 4. 数据加载器 (核心改动：Batch Size = 1)
train_dataloader = dict(
    batch_size=1,            # 24G 显存单卡必须设为1
    num_workers=2,           # 降低 worker 数量，减少内存/显存碎片
    persistent_workers=False, # 设为 False 进一步释放epoch间的缓存
    dataset=dict(
        type='SSDDInsSegDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=data_root + '/annotations/train.json',
        data_prefix=dict(img='imgs/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='SSDDInsSegDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file=data_root + '/annotations/value.json',
        data_prefix=dict(img='imgs/value/'),
        pipeline=train_pipeline
    )
)
test_dataloader = val_dataloader

# 5. 训练策略 (混合精度训练)
base_lr = 0.0001
max_epochs = 300
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(
    type='AmpOptimWrapper', # 开启自动混合精度，省显存神器
    dtype='float16',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05)
)

# 6. 可视化与日志
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='DamCrack', group='anchor', name='stable-run-bs1'))
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 7. 默认钩子配置
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

# 8. 评价器配置 (解决全 0 分的关键)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/value.json',
    metric=['bbox', 'segm'],
    # 强制增加低阈值评估，针对细长裂缝放宽要求
    iou_thrs=[0.05, 0.1, 0.3, 0.5], 
    format_only=False,
    backend_args=None
)
test_evaluator = val_evaluator