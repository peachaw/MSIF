"""
工具函数: 指标计算、早停机制、随机种子设置
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# ======================== 随机种子 ========================

def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ======================== 评估指标 ========================

def compute_metrics(y_true, y_pred):
    """
    计算准确率和加权 F1 分数。
    
    Args:
        y_true: 真实标签 (list or array)
        y_pred: 预测标签 (list or array)
    Returns:
        dict: {"accuracy": float, "weighted_f1": float}
    """
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="weighted") * 100
    return {"accuracy": acc, "weighted_f1": f1}


# ======================== 早停机制 ========================

class EarlyStopping:
    """
    基于验证集指标的早停机制。
    
    Args:
        patience: 容忍的 epoch 数
        metric_name: 监控的指标名称
        mode: 'max' 或 'min'
        save_path: 最佳模型保存路径
    """
    
    def __init__(self, patience=10, metric_name="weighted_f1",
                 mode="max", save_path="best_model.pt"):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.save_path = save_path
        
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, metrics, model):
        score = metrics[self.metric_name]
        
        if self.best_score is None:
            self.best_score = score
            self._save_model(model)
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self._save_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def _is_improvement(self, score):
        if self.mode == "max":
            return score > self.best_score
        else:
            return score < self.best_score
    
    def _save_model(self, model):
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
    
    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_path, map_location="cpu"))
        return model


# ======================== 日志辅助 ========================

class AverageMeter:
    """计算并存储均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_metrics(metrics):
    """格式化指标为可打印字符串"""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}: {v:.2f}")
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)
