from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class DamCrackDataset(CocoDataset):
    METAINFO = {
        'classes': ('crack',), # 只有一类：裂缝
        'palette': [(220, 20, 60)] # 可视化用的红色
    }