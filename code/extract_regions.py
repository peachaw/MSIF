"""
区域特征提取脚本

使用 torchvision 的 Faster R-CNN (ResNet-50-FPN) 提取图像区域特征。
对每张图像检测 top-m 个区域，并提取 RoI 池化后的特征向量。

注意:
  - 本脚本使用 torchvision 预训练的 Faster R-CNN (COCO 预训练)
  - 论文中使用的是 Visual Genome 预训练的 Faster R-CNN
  - 如需精确复现，请使用 Bottom-Up Attention 模型:
    https://github.com/airsplay/py-bottom-up-attention

用法:
  python extract_regions.py \
    --data_dir ./data/MVSA_Single/data \
    --output_dir ./data/region_features/MVSA_Single \
    --num_regions 36

输出:
  每张图像生成一个 .npy 文件，shape = (m, feat_dim)
  feat_dim = 1024 (torchvision FPN) 或 2048 (Bottom-Up Attention)
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import roi_align


def extract_with_torchvision(image_dir, output_dir, num_regions=36,
                              device="cuda", target_feat_dim=2048):
    """
    使用 torchvision Faster R-CNN 提取区域特征。
    
    流程:
      1. 将图像送入 Faster R-CNN 获取检测框
      2. 使用 backbone 提取特征图
      3. 对检测框执行 RoI Align 获取区域特征
      4. 通过自适应平均池化 + 线性投影得到目标维度
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    model.to(device)
    
    # 获取 backbone 用于特征图提取
    backbone = model.backbone
    
    # 图像预处理 (Faster R-CNN 自带预处理，但我们需要手动控制)
    preprocess = weights.transforms()
    
    # 特征投影: 将 FPN 特征映射到目标维度
    # FPN 输出通道数为 256，RoI pooling 后是 256 * 7 * 7 = 12544
    # 我们使用 box_head 的输出 (1024-dim)，再投影到 target_feat_dim
    proj_layer = nn.Linear(1024, target_feat_dim).to(device)
    nn.init.xavier_uniform_(proj_layer.weight)
    
    # 收集图像文件
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = []
    
    if os.path.isdir(image_dir):
        for item in os.listdir(image_dir):
            item_path = os.path.join(image_dir, item)
            if os.path.isfile(item_path):
                _, ext = os.path.splitext(item)
                if ext.lower() in extensions:
                    sample_id = os.path.splitext(item)[0]
                    image_files.append((sample_id, item_path))
            elif os.path.isdir(item_path):
                # 子目录结构: data/{id}/{id}.jpg
                for f in os.listdir(item_path):
                    _, ext = os.path.splitext(f)
                    if ext.lower() in extensions:
                        image_files.append((item, os.path.join(item_path, f)))
                        break
    
    print(f"找到 {len(image_files)} 张图像")
    
    for sample_id, image_path in tqdm(image_files, desc="提取区域特征"):
        output_path = os.path.join(output_dir, f"{sample_id}.npy")
        if os.path.exists(output_path):
            continue
        
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 获取检测结果
                detections = model(img_tensor)[0]
                boxes = detections["boxes"]    # (N, 4)
                scores = detections["scores"]  # (N,)
                
                # 选择 top-m 个高置信度区域
                if len(boxes) == 0:
                    # 无检测结果，使用整张图作为唯一区域
                    h, w = img_tensor.shape[2:]
                    boxes = torch.tensor([[0, 0, w, h]], dtype=torch.float32).to(device)
                    scores = torch.tensor([1.0]).to(device)
                
                if len(boxes) >= num_regions:
                    top_indices = scores.argsort(descending=True)[:num_regions]
                    boxes = boxes[top_indices]
                
                # 获取特征图
                features = backbone(img_tensor)
                
                # 使用最高分辨率的特征图 ('0' 层)
                feat_map = features["0"]  # (1, 256, H/4, W/4)
                
                # RoI Align
                # 需要在 boxes 前面加上 batch index
                batch_indices = torch.zeros(len(boxes), 1, device=device)
                rois = torch.cat([batch_indices, boxes], dim=1)  # (N, 5)
                
                roi_features = roi_align(
                    feat_map, rois,
                    output_size=(7, 7),
                    spatial_scale=1.0 / 4.0,  # 特征图步幅
                    aligned=True,
                )  # (N, 256, 7, 7)
                
                # 展平并通过 box_head 的两个全连接层
                roi_features = roi_features.flatten(start_dim=1)  # (N, 256*7*7)
                
                # 使用模型的 box_head 获取更好的表示
                roi_features_repr = model.roi_heads.box_head(
                    roi_features.view(-1, 256, 7, 7)
                )  # (N, 1024)
                
                # 投影到目标维度
                region_feats = proj_layer(roi_features_repr)  # (N, target_feat_dim)
                
                # 填充到 num_regions
                n_detected = region_feats.shape[0]
                if n_detected < num_regions:
                    pad = torch.zeros(
                        num_regions - n_detected, target_feat_dim,
                        device=device
                    )
                    region_feats = torch.cat([region_feats, pad], dim=0)
                else:
                    region_feats = region_feats[:num_regions]
                
                # 保存
                np.save(output_path, region_feats.cpu().numpy())
        
        except Exception as e:
            print(f"  [跳过] {sample_id}: {e}")
            # 保存零特征以避免训练时报错
            np.save(
                output_path,
                np.zeros((num_regions, target_feat_dim), dtype=np.float32)
            )
    
    print(f"区域特征已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="提取图像区域特征")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="图像数据目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num_regions", type=int, default=36,
                        help="每张图像提取的区域数量")
    parser.add_argument("--target_feat_dim", type=int, default=2048,
                        help="目标特征维度")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    extract_with_torchvision(
        image_dir=args.data_dir,
        output_dir=args.output_dir,
        num_regions=args.num_regions,
        device=device,
        target_feat_dim=args.target_feat_dim,
    )


if __name__ == "__main__":
    main()
