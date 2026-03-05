"""
MVSA 数据集加载与预处理

支持 MVSA-Single 和 MVSA-Multiple 两个数据集。
数据预处理规则（论文 4.1 节）:
  - 图像标签与文本标签一致 → 使用该标签
  - 一个为正/负，另一个为中性 → 使用正/负
  - 一个为正，另一个为负 → 丢弃该样本

预期数据目录结构:
  data/
  ├── MVSA_Single/
  │   ├── data/           # 包含 {id}.jpg 和 {id}.txt
  │   └── labelResultAll.txt
  ├── MVSA_Multiple/
  │   ├── data/
  │   └── labelResultAll.txt
  └── region_features/    # 预提取的区域特征
      ├── MVSA_Single/
      │   └── {id}.npy    # 每个文件 shape: (m, 2048)
      └── MVSA_Multiple/
          └── {id}.npy
"""

import os
import re
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ======================== 标签映射 ========================

LABEL_MAP = {"positive": 0, "neutral": 1, "negative": 2}
LABEL_NAMES = ["positive", "neutral", "negative"]


# ======================== 数据预处理 ========================

def resolve_label(text_label, image_label):
    """
    根据论文 4.1 节的规则合并图文标签。
    返回: 合并后的标签字符串，或 None（表示丢弃）
    """
    text_label = text_label.strip().lower()
    image_label = image_label.strip().lower()
    
    if text_label == image_label:
        return text_label
    
    # 一个正/负 + 另一个中性 → 使用正/负
    if text_label == "neutral":
        return image_label
    if image_label == "neutral":
        return text_label
    
    # 正 vs 负 → 丢弃
    return None


def load_mvsa_single_labels(label_file):
    """
    加载 MVSA-Single 标签文件。
    支持两种格式:
      格式A: id \\t text_label,image_label   (实际 MVSA 格式)
      格式B: id \\t text_label \\t image_label
    """
    samples = []
    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            
            sample_id = parts[0].strip()
            
            # 跳过标题行 (如 "ID\ttext,image")
            if sample_id.upper() == "ID":
                continue
            
            # 解析标签
            if len(parts) == 2:
                # 格式A: "id\ttext_label,image_label"
                label_parts = parts[1].strip().split(",")
                if len(label_parts) < 2:
                    continue
                text_label = label_parts[0].strip().lower()
                image_label = label_parts[1].strip().lower()
            elif len(parts) >= 3:
                # 格式B: "id\ttext_label\timage_label"
                text_label = parts[1].strip().lower()
                image_label = parts[2].strip().lower()
            else:
                continue
            
            if text_label not in LABEL_MAP or image_label not in LABEL_MAP:
                continue
            
            label = resolve_label(text_label, image_label)
            if label is not None:
                samples.append({"id": sample_id, "label": label})
    
    return samples


def load_mvsa_multiple_labels(label_file):
    """
    加载 MVSA-Multiple 标签文件。
    格式: id \\t text_label1 \\t image_label1 \\t text_label2 \\t image_label2 \\t text_label3 \\t image_label3
    先对文本标签多数投票，再对图像标签多数投票，最后合并。
    """
    from collections import Counter
    
    samples = []
    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                parts = line.split()
            if len(parts) < 7:
                continue
            
            sample_id = parts[0].strip()
            
            text_labels = [parts[i].strip().lower() for i in range(1, 7, 2)]
            image_labels = [parts[i].strip().lower() for i in range(2, 7, 2)]
            
            # 过滤无效标签
            text_labels = [l for l in text_labels if l in LABEL_MAP]
            image_labels = [l for l in image_labels if l in LABEL_MAP]
            
            if not text_labels or not image_labels:
                continue
            
            # 多数投票
            text_label = Counter(text_labels).most_common(1)[0][0]
            image_label = Counter(image_labels).most_common(1)[0][0]
            
            label = resolve_label(text_label, image_label)
            if label is not None:
                samples.append({"id": sample_id, "label": label})
    
    return samples


def load_text(text_path):
    """加载文本文件内容"""
    encodings = ["utf-8", "latin-1", "gbk", "ascii"]
    for enc in encodings:
        try:
            with open(text_path, "r", encoding=enc) as f:
                text = f.read().strip()
            # 清理文本：去除 URL、@mention、多余空白
            text = re.sub(r"http\S+|www\.\S+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""


# ======================== 数据集类 ========================

class MVSADataset(Dataset):
    """
    MVSA 数据集
    
    Args:
        samples:        list of dict, 每项包含 'id' 和 'label'
        data_dir:       图文数据根目录 (包含 {id}.jpg, {id}.txt 或 {id}/ 子目录)
        region_feat_dir: 预提取区域特征目录 (包含 {id}.npy)
        tokenizer:      BERT 分词器
        max_text_len:   文本最大长度
        image_size:     图像输入尺寸
        num_regions:    区域数量 m
        region_feat_dim: 区域特征维度
        transform:      图像变换 (如为 None 则使用默认变换)
    """
    
    def __init__(self, samples, data_dir, region_feat_dir,
                 tokenizer, max_text_len=140, image_size=224,
                 num_regions=36, region_feat_dim=2048,
                 transform=None):
        self.samples = samples
        self.data_dir = data_dir
        self.region_feat_dir = region_feat_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.num_regions = num_regions
        self.region_feat_dim = region_feat_dim
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
    def _find_image(self, sample_id):
        """查找图像文件（支持多种目录结构和扩展名）"""
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        
        # 结构1: data_dir/{id}.ext
        for ext in extensions:
            path = os.path.join(self.data_dir, f"{sample_id}{ext}")
            if os.path.exists(path):
                return path
        
        # 结构2: data_dir/{id}/{id}.ext 或 data_dir/{id}/image.ext
        subdir = os.path.join(self.data_dir, str(sample_id))
        if os.path.isdir(subdir):
            for ext in extensions:
                for name in [sample_id, "image", str(sample_id)]:
                    path = os.path.join(subdir, f"{name}{ext}")
                    if os.path.exists(path):
                        return path
            # 尝试找第一个图像文件
            for f in os.listdir(subdir):
                if any(f.lower().endswith(e) for e in extensions):
                    return os.path.join(subdir, f)
        
        return None
    
    def _find_text(self, sample_id):
        """查找文本文件"""
        # 结构1: data_dir/{id}.txt
        path = os.path.join(self.data_dir, f"{sample_id}.txt")
        if os.path.exists(path):
            return path
        
        # 结构2: data_dir/{id}/{id}.txt 或 data_dir/{id}/text.txt
        subdir = os.path.join(self.data_dir, str(sample_id))
        if os.path.isdir(subdir):
            for name in [f"{sample_id}.txt", "text.txt"]:
                path = os.path.join(subdir, name)
                if os.path.exists(path):
                    return path
            for f in os.listdir(subdir):
                if f.endswith(".txt"):
                    return os.path.join(subdir, f)
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["id"]
        label = LABEL_MAP[sample["label"]]
        
        # ---- 加载文本 ----
        text_path = self._find_text(sample_id)
        text = load_text(text_path) if text_path else ""
        if not text:
            text = "empty"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)          # (max_text_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (max_text_len,)
        
        # ---- 加载图像 ----
        image_path = self._find_image(sample_id)
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
            except Exception:
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)
        
        # ---- 加载区域特征 ----
        region_path = os.path.join(self.region_feat_dir, f"{sample_id}.npy")
        if os.path.exists(region_path):
            region_features = np.load(region_path)
            # 确保形状为 (m, feat_dim)
            if len(region_features.shape) == 1:
                region_features = region_features.reshape(1, -1)
            # 截断或填充到 num_regions
            m, d = region_features.shape
            if m >= self.num_regions:
                region_features = region_features[:self.num_regions]
            else:
                pad = np.zeros((self.num_regions - m, d), dtype=np.float32)
                region_features = np.concatenate([region_features, pad], axis=0)
            region_features = torch.tensor(region_features, dtype=torch.float32)
        else:
            # 无预提取特征 → 零填充 (需要先运行 extract_regions.py)
            region_features = torch.zeros(self.num_regions, self.region_feat_dim)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "region_features": region_features,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
        }


# ======================== 路径辅助 ========================

def _find_dataset_dir(data_dir, *name_variants):
    """
    自动检测数据集目录，支持多种嵌套结构:
      data_dir/MVSA_Single/data/
      data_dir/MVSA-Single/MVSA_Single/data/
      data_dir/MVSA_Single/MVSA_Single/data/
    返回包含 labelResultAll.txt 的目录路径。
    """
    candidates = []
    for name in name_variants:
        # 直接: data_dir/name/
        candidates.append(os.path.join(data_dir, name))
        # 嵌套: data_dir/name/name_inner/
        inner_dir = os.path.join(data_dir, name)
        if os.path.isdir(inner_dir):
            for sub in os.listdir(inner_dir):
                sub_path = os.path.join(inner_dir, sub)
                if os.path.isdir(sub_path):
                    candidates.append(sub_path)
    
    for path in candidates:
        label_file = os.path.join(path, "labelResultAll.txt")
        if os.path.exists(label_file):
            print(f"[数据集] 找到数据目录: {path}")
            return path
    
    # 回退: 返回第一个候选路径（可能不存在，后续会报错）
    fallback = os.path.join(data_dir, name_variants[0])
    print(f"[警告] 未找到 labelResultAll.txt，使用默认路径: {fallback}")
    return fallback


# ======================== 数据集构建 ========================

def build_datasets(config):
    """
    构建训练/验证/测试数据集。
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    
    # 确定数据路径 (自动检测目录结构)
    if config.dataset == "MVSA-Single":
        dataset_dir = _find_dataset_dir(config.data_dir, "MVSA_Single", "MVSA-Single")
        data_dir = os.path.join(dataset_dir, "data")
        label_file = os.path.join(dataset_dir, "labelResultAll.txt")
        region_dir = os.path.join(config.region_feat_dir, "MVSA_Single")
        samples = load_mvsa_single_labels(label_file)
    elif config.dataset == "MVSA-Multiple":
        dataset_dir = _find_dataset_dir(config.data_dir, "MVSA_Multiple", "MVSA-Multiple")
        data_dir = os.path.join(dataset_dir, "data")
        label_file = os.path.join(dataset_dir, "labelResultAll.txt")
        region_dir = os.path.join(config.region_feat_dir, "MVSA_Multiple")
        samples = load_mvsa_multiple_labels(label_file)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    print(f"[数据集] {config.dataset}: 共加载 {len(samples)} 个有效样本")
    
    # 打印类别分布
    from collections import Counter
    dist = Counter(s["label"] for s in samples)
    for label_name in LABEL_NAMES:
        print(f"  {label_name}: {dist.get(label_name, 0)}")
    
    # 按 80/10/10 划分 (使用固定种子 42 确保一致性)
    labels_for_split = [s["label"] for s in samples]
    train_samples, temp_samples = train_test_split(
        samples, test_size=0.2, random_state=42, stratify=labels_for_split
    )
    temp_labels = [s["label"] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"  训练集: {len(train_samples)}, 验证集: {len(val_samples)}, 测试集: {len(test_samples)}")
    
    # 构建 Dataset
    common_kwargs = dict(
        data_dir=data_dir,
        region_feat_dir=region_dir,
        tokenizer=tokenizer,
        max_text_len=config.max_text_len,
        image_size=config.image_size,
        num_regions=config.num_regions,
        region_feat_dim=config.region_feat_dim,
    )
    
    train_dataset = MVSADataset(train_samples, **common_kwargs)
    val_dataset = MVSADataset(val_samples, **common_kwargs)
    test_dataset = MVSADataset(test_samples, **common_kwargs)
    
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(config, train_dataset, val_dataset, test_dataset):
    """构建 DataLoader"""
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
