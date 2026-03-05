"""
MSIF 训练脚本

支持:
  - 单次训练（指定 --seed）
  - 多种子训练（--multi_seed，自动使用 --seeds 列表运行并汇总均值±标准差）
  - 消融实验（通过 --no_alignment, --no_fusion, --no_gating 等开关控制）

用法示例:
  # 默认训练 (MVSA-Single, seed=42)
  python train.py --dataset MVSA-Single

  # 多种子训练
  python train.py --dataset MVSA-Single --multi_seed

  # 消融: 移除对齐模块
  python train.py --dataset MVSA-Single --no_alignment --multi_seed

  # 消融: 移除门控
  python train.py --dataset MVSA-Single --no_gating --multi_seed

  # 不同区域数量
  python train.py --dataset MVSA-Single --num_regions 10 --multi_seed
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import get_config, DATASET_CONFIGS
from model import MSIF
from dataset import build_datasets, build_dataloaders
from utils import set_seed, compute_metrics, EarlyStopping, AverageMeter, format_metrics


# ======================== 训练一个 epoch ========================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        region_features = batch["region_features"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, image, region_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), labels.size(0))
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
        
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = loss_meter.avg
    return metrics


# ======================== 验证/测试 ========================

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        region_features = batch["region_features"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(input_ids, attention_mask, image, region_features)
        loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), labels.size(0))
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = loss_meter.avg
    return metrics


# ======================== 单次运行 ========================

def run_single(config, seed):
    """使用指定随机种子进行一次完整的训练-验证-测试流程"""
    print(f"\n{'='*60}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")
    
    # 构建数据集
    train_dataset, val_dataset, test_dataset = build_datasets(config)
    train_loader, val_loader, test_loader = build_dataloaders(
        config, train_dataset, val_dataset, test_dataset
    )
    
    # 构建模型
    model = MSIF(config).to(device)
    total_params = model.count_parameters()
    print(f"[模型] 可训练参数量: {total_params:,}")
    
    # 损失函数、优化器、学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=config.lr_decay,
        patience=config.lr_patience, verbose=True,
    )
    
    # 早停
    save_path = os.path.join(
        config.save_dir, config.dataset,
        f"seed_{seed}", "best_model.pt"
    )
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        metric_name="weighted_f1",
        mode="max",
        save_path=save_path,
    )
    
    # 训练循环
    best_val_metrics = None
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  [Train] {format_metrics(train_metrics)}")
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"  [Val]   {format_metrics(val_metrics)}")
        
        # 学习率调度
        scheduler.step(val_metrics["weighted_f1"])
        
        # 早停检查
        early_stopping(val_metrics, model)
        if early_stopping.best_score == val_metrics["weighted_f1"]:
            best_val_metrics = val_metrics.copy()
            print(f"  ★ 新最优验证 F1: {val_metrics['weighted_f1']:.2f}")
        
        if early_stopping.should_stop:
            print(f"\n  早停触发 (patience={config.early_stop_patience})")
            break
    
    # 加载最佳模型并测试
    model = early_stopping.load_best_model(model)
    model.to(device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n[Test] {format_metrics(test_metrics)}")
    
    return {
        "seed": seed,
        "best_val": best_val_metrics,
        "test": test_metrics,
        "num_params": total_params,
    }


# ======================== 多种子运行 ========================

def run_multi_seed(config):
    """使用多个随机种子运行，汇总均值±标准差"""
    all_results = []
    
    for seed in config.seeds:
        result = run_single(config, seed)
        all_results.append(result)
    
    # 汇总
    test_accs = [r["test"]["accuracy"] for r in all_results]
    test_f1s = [r["test"]["weighted_f1"] for r in all_results]
    
    print(f"\n{'='*60}")
    print(f"  多种子结果汇总 ({len(config.seeds)} 次运行)")
    print(f"{'='*60}")
    print(f"  Seeds: {config.seeds}")
    print(f"  Test Accuracy: {np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}")
    print(f"  Test F1:       {np.mean(test_f1s):.2f} ± {np.std(test_f1s):.2f}")
    print(f"  参数量:        {all_results[0]['num_params']:,}")
    
    # 保存结果
    summary = {
        "dataset": config.dataset,
        "config": {
            "use_alignment": config.use_alignment,
            "use_fusion": config.use_fusion,
            "use_gating": config.use_gating,
            "use_bidirectional": config.use_bidirectional,
            "use_bigru": config.use_bigru,
            "num_regions": config.num_regions,
        },
        "seeds": config.seeds,
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "per_seed_results": [
            {
                "seed": r["seed"],
                "test_accuracy": r["test"]["accuracy"],
                "test_f1": r["test"]["weighted_f1"],
            }
            for r in all_results
        ],
        "num_params": all_results[0]["num_params"],
    }
    
    # 生成实验标签
    ablation_tag = "full"
    if not config.use_alignment:
        ablation_tag = "wo_align"
    elif not config.use_fusion:
        ablation_tag = "wo_fusion"
    elif not config.use_gating:
        ablation_tag = "wo_gating"
    elif not config.use_bidirectional:
        ablation_tag = "wo_bidirectional"
    elif not config.use_bigru:
        ablation_tag = "wo_bigru"
    
    log_dir = os.path.join(config.log_dir, config.dataset)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_dir,
        f"{ablation_tag}_m{config.num_regions}_{timestamp}.json"
    )
    with open(log_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  结果已保存至: {log_file}")
    
    return summary


# ======================== 主入口 ========================

def main():
    config = get_config()
    
    # 应用数据集预设超参
    if config.dataset in DATASET_CONFIGS:
        preset = DATASET_CONFIGS[config.dataset]
        for k, v in preset.items():
            if f"--{k}" not in sys.argv:
                setattr(config, k, v)
    
    print(f"[配置] 数据集: {config.dataset}")
    print(f"[配置] 对齐模块: {config.use_alignment}")
    print(f"[配置] 融合模块: {config.use_fusion}")
    print(f"[配置] 门控机制: {config.use_gating}")
    print(f"[配置] 双向一致性: {config.use_bidirectional}")
    print(f"[配置] BiGRU: {config.use_bigru}")
    print(f"[配置] 区域数量: {config.num_regions}")
    print(f"[配置] 批大小: {config.batch_size}")
    print(f"[配置] 学习率: {config.lr}")
    
    if config.multi_seed:
        run_multi_seed(config)
    else:
        run_single(config, config.seed)


if __name__ == "__main__":
    main()
