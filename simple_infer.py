import torch
import mmcv
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

# --- 0. 显存清理 ---
torch.cuda.empty_cache()

# --- 1. 路径配置 ---
img_path = '/root/autodl-tmp/RSPrompter/data/DamCrack/imgs/value/2363_jpg.rf.d318477230b425c21cc19efaaa8fa12a.jpg'
config_file = '/root/autodl-tmp/RSPrompter/configs/rsprompter/samdet_damcrack_3.py'
checkpoint_file = '/root/autodl-tmp/RSPrompter/work_dirs/rsprompter/damcrack_anchor_final/best_coco_segm_mAP_epoch_275.pth'
out_file = '/root/autodl-tmp/RSPrompter/work_dirs/result/prediction_2363_segm.jpg'

os.makedirs(os.path.dirname(out_file), exist_ok=True)

# 2. 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 3. 读取图片并获取真实宽度
img = mmcv.imread(img_path)
real_h, real_w = img.shape[:2] # 获取这张图真实的宽高
print(f"检测到图片原始尺寸: {real_w}x{real_h}")

# 4. 推理
print("正在进行推理...")
with torch.no_grad():
    result = inference_detector(model, img_path)


# 5. 可视化
visualizer = DetLocalVisualizer(name='final_visualizer')
if hasattr(model, 'dataset_meta'):
    visualizer.dataset_meta = model.dataset_meta

img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')

visualizer.add_datasample(
    'result',
    img_rgb,
    data_sample=result,
    draw_gt=False,
    show=False,
    out_file=out_file,
    pred_score_thr=0.3
)

print(f"\n🎉 修正版推理完成！请查看: {out_file}")