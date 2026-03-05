"""
MSIF 模型配置文件
所有超参数和实验设置均集中管理于此。
"""
import argparse


def get_config():
    parser = argparse.ArgumentParser(description="MSIF: Multimodal Sentiment Interaction and Fusion")

    # ======================== 数据相关 ========================
    parser.add_argument("--dataset", type=str, default="MVSA-Single",
                        choices=["MVSA-Single", "MVSA-Multiple"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据集根目录")
    parser.add_argument("--region_feat_dir", type=str, default="./data/region_features",
                        help="预提取的区域特征目录")
    parser.add_argument("--max_text_len", type=int, default=140,
                        help="文本最大长度")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像输入尺寸")
    parser.add_argument("--num_regions", type=int, default=36,
                        help="每张图像提取的区域数量 m")

    # ======================== 模型结构 ========================
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="预训练 BERT 模型名称")
    parser.add_argument("--freeze_bert", action="store_true", default=False,
                        help="是否冻结 BERT 参数")
    parser.add_argument("--gru_hidden", type=int, default=256,
                        help="BiGRU 每个方向的隐藏层维度")
    parser.add_argument("--region_feat_dim", type=int, default=2048,
                        help="区域特征原始维度（Faster R-CNN 输出）")
    parser.add_argument("--d_align", type=int, default=768,
                        help="对齐特征维度（需与 BERT 隐藏层维度一致）")
    parser.add_argument("--d_attn", type=int, default=256,
                        help="融合注意力模块中的投影维度")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="情感类别数（positive/neutral/negative）")

    # ======================== 消融开关 ========================
    parser.add_argument("--use_alignment", action="store_true", default=True,
                        help="是否使用图文特征对齐模块")
    parser.add_argument("--no_alignment", dest="use_alignment", action="store_false",
                        help="移除图文特征对齐模块（消融实验）")
    parser.add_argument("--use_fusion", action="store_true", default=True,
                        help="是否使用融合注意力模块")
    parser.add_argument("--no_fusion", dest="use_fusion", action="store_false",
                        help="移除融合注意力模块（消融实验）")
    parser.add_argument("--use_gating", action="store_true", default=True,
                        help="是否使用门控机制")
    parser.add_argument("--no_gating", dest="use_gating", action="store_false",
                        help="移除门控机制（消融实验）")
    parser.add_argument("--use_bidirectional", action="store_true", default=True,
                        help="是否使用双向一致性约束")
    parser.add_argument("--no_bidirectional", dest="use_bidirectional", action="store_false",
                        help="移除双向一致性约束（消融实验）")
    parser.add_argument("--use_bigru", action="store_true", default=True,
                        help="是否使用 BiGRU")
    parser.add_argument("--no_bigru", dest="use_bigru", action="store_false",
                        help="移除 BiGRU（消融实验）")

    # ======================== 训练参数 ========================
    parser.add_argument("--batch_size", type=int, default=32,
                        help="训练批大小")
    parser.add_argument("--epochs", type=int, default=50,
                        help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=2e-5,
                        help="权重衰减")
    parser.add_argument("--lr_decay", type=float, default=0.1,
                        help="学习率衰减因子")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="学习率衰减的 patience（ReduceLROnPlateau）")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="早停 patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024],
                        help="多次运行使用的随机种子列表")
    parser.add_argument("--multi_seed", action="store_true", default=False,
                        help="是否使用多个种子运行（报告均值±标准差）")

    # ======================== 其他 ========================
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备 cuda/cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志保存目录")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")

    config = parser.parse_args()
    return config


# 预设配置：两个数据集的推荐超参数
DATASET_CONFIGS = {
    "MVSA-Single": {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 2e-5,
    },
    "MVSA-Multiple": {
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 2e-5,
    },
}
