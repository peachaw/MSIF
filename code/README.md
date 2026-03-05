# MSIF: Multimodal Sentiment Interaction and Fusion

## 项目结构

```
code/
├── config.py            # 所有超参数与消融开关
├── model.py             # MSIF 完整模型（TextEncoder, CrossModalAlignment, FusionAttention）
├── dataset.py           # MVSA 数据集加载与预处理
├── train.py             # 训练/验证/测试流程（支持多种子运行）
├── extract_regions.py   # 图像区域特征提取
├── utils.py             # 指标计算、早停、种子设置
├── requirements.txt     # 依赖
└── README.md
```

## 环境配置

```bash
pip install -r requirements.txt
```

需要: Python >= 3.8, PyTorch >= 1.12, CUDA (推荐)

## 数据准备

### 1. 下载 MVSA 数据集

从 MVSA 官方获取数据集并解压到 `data/` 目录：

```
data/
├── MVSA_Single/
│   ├── data/           # 图像和文本文件
│   └── labelResultAll.txt
└── MVSA_Multiple/
    ├── data/
    └── labelResultAll.txt
```

### 2. 提取区域特征

```bash
# MVSA-Single
python extract_regions.py \
  --data_dir ./data/MVSA_Single/data \
  --output_dir ./data/region_features/MVSA_Single \
  --num_regions 36

# MVSA-Multiple
python extract_regions.py \
  --data_dir ./data/MVSA_Multiple/data \
  --output_dir ./data/region_features/MVSA_Multiple \
  --num_regions 36
```

## 训练

### 基本训练

```bash
# MVSA-Single (单次, seed=42)
python train.py --dataset MVSA-Single --seed 42

# MVSA-Multiple
python train.py --dataset MVSA-Multiple --seed 42
```

### 多种子训练（报告均值±标准差）

```bash
python train.py --dataset MVSA-Single --multi_seed
python train.py --dataset MVSA-Multiple --multi_seed
```

默认使用 5 个种子: 42, 123, 456, 789, 1024

## 消融实验

### 模块级消融

```bash
# w/o Alignment (移除对齐模块)
python train.py --dataset MVSA-Single --no_alignment --multi_seed

# w/o Fusion (移除融合注意力模块)
python train.py --dataset MVSA-Single --no_fusion --multi_seed
```

### 组件级消融

```bash
# w/o Gating (移除门控机制)
python train.py --dataset MVSA-Single --no_gating --multi_seed

# w/o Bidirectional Consistency (移除双向一致性约束)
python train.py --dataset MVSA-Single --no_bidirectional --multi_seed

# w/o BiGRU (移除 BiGRU)
python train.py --dataset MVSA-Single --no_bigru --multi_seed
```

### 区域数量敏感性

```bash
python train.py --dataset MVSA-Single --num_regions 10 --multi_seed
python train.py --dataset MVSA-Single --num_regions 20 --multi_seed
python train.py --dataset MVSA-Single --num_regions 36 --multi_seed
python train.py --dataset MVSA-Single --num_regions 50 --multi_seed
```

## 完整实验脚本

运行所有实验（两个数据集 × 所有消融变体）：

```bash
#!/bin/bash
# run_all_experiments.sh

DATASETS=("MVSA-Single" "MVSA-Multiple")

for DS in "${DATASETS[@]}"; do
  echo "===== Dataset: $DS ====="

  # 完整模型
  python train.py --dataset "$DS" --multi_seed

  # 模块级消融
  python train.py --dataset "$DS" --no_alignment --multi_seed
  python train.py --dataset "$DS" --no_fusion --multi_seed

  # 组件级消融
  python train.py --dataset "$DS" --no_gating --multi_seed
  python train.py --dataset "$DS" --no_bidirectional --multi_seed
  python train.py --dataset "$DS" --no_bigru --multi_seed

  # 区域数量
  for M in 10 20 50; do
    python train.py --dataset "$DS" --num_regions $M --multi_seed
  done
done
```

## 结果

实验结果会自动保存为 JSON 文件到 `logs/` 目录，格式为：

```json
{
  "dataset": "MVSA-Single",
  "test_accuracy_mean": 76.72,
  "test_accuracy_std": 0.35,
  "test_f1_mean": 75.90,
  "test_f1_std": 0.42,
  "per_seed_results": [...]
}
```

## 超参数

| 参数 | MVSA-Single | MVSA-Multiple |
|---|---|---|
| Image Size | 224 | 224 |
| Text Length | 140 | 140 |
| Batch Size | 32 | 64 |
| Optimizer | Adam | Adam |
| Learning Rate | 1e-4 | 1e-4 |
| Weight Decay | 2e-5 | 2e-5 |
| LR Decay | 0.1 | 0.1 |
| Epochs | 50 | 50 |
| Early Stop Patience | 10 | 10 |
| Num Regions (m) | 36 | 36 |
| GRU Hidden | 256 | 256 |
| Attention Dim | 256 | 256 |
