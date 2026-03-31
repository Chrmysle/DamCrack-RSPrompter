import torch
import mmcv
import os
import shutil
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer

# --- 0. 显存清理 ---
torch.cuda.empty_cache()

# --- 1. 路径配置 ---
target_dir = '/root/autodl-tmp/RSPrompter/work_dirs/result'
config_file = '/root/autodl-tmp/RSPrompter/configs/rsprompter/samdet_damcrack_3.py'
checkpoint_file = '/root/autodl-tmp/RSPrompter/work_dirs/rsprompter/damcrack_anchor_final/best_coco_segm_mAP_epoch_275.pth'

# --- 2. 初始化模型 (放在循环外，只加载一次) ---
print("正在加载模型到 GPU...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 初始化可视化器
visualizer = DetLocalVisualizer(name='final_visualizer')
if hasattr(model, 'dataset_meta'):
    visualizer.dataset_meta = model.dataset_meta

# --- 3. 获取目标文件夹中的所有图片 ---
# 支持的图片后缀
valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
image_files = []

# 遍历 result 目录，只筛选图片文件（跳过已经创建的文件夹）
for f in os.listdir(target_dir):
    file_path = os.path.join(target_dir, f)
    if os.path.isfile(file_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            image_files.append(f)

# 按照文件名排序，保证每次处理顺序一致
image_files.sort()

print(f"检测到 {len(image_files)} 张待处理图片，开始批量推理...")

# --- 4. 批量推理与保存 ---
for idx, img_name in enumerate(image_files):
    # 原始图片路径
    img_path = os.path.join(target_dir, img_name)
    
    # 获取不带后缀的基础文件名
    base_name = os.path.splitext(img_name)[0]
    
    # 创建独立的输出文件夹，格式例如: 001_2141_jpg
    out_folder_name = f"{idx+1:03d}_{base_name}"
    out_folder_path = os.path.join(target_dir, out_folder_name)
    os.makedirs(out_folder_path, exist_ok=True)
    
    # 定义原图备份路径和预测结果保存路径
    orig_save_path = os.path.join(out_folder_path, f"original_{img_name}")
    pred_save_path = os.path.join(out_folder_path, f"prediction_{img_name}")
    
    # 拷贝原图到专属文件夹
    shutil.copy(img_path, orig_save_path)
    
    # 读取图片
    img = mmcv.imread(img_path)
    
    # 推理
    with torch.no_grad():
        result = inference_detector(model, img_path)

    # 可视化并保存预测图
    img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img_rgb,
        data_sample=result,
        draw_gt=False,
        show=False,
        out_file=pred_save_path,
        pred_score_thr=0.3
    )
    
    print(f"[{idx+1}/{len(image_files)}] 完成推理: {img_name} -> 已存入 {out_folder_name}/")

print("\n🎉 所有图片批量推理完成！")